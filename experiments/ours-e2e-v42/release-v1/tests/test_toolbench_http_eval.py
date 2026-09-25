import ast
import copy
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from test_toolbench_agent import tools
from latent_register.toolbench_agent import Decision, ExecutionResult, GenerationFailure, run_serial_agent
from latent_register.toolbench_data import FINISH, call_history, observation
from latent_register.toolbench_eval_format import convert_trace, evaluator_names
from latent_register.toolbench_http import create_executor, wire_bindings


@pytest.fixture
def bindings():
    return {"city/search": {"category_name": "Location", "tool_name": "City", "api_name": "search"},
            "weather/forecast": {"category_name": "Weather", "tool_name": "Weather", "api_name": "forecast"}}


@pytest.fixture
def server():
    records, reply = [], {"status": 200, "body": {"error": "", "response": {"city_id": 917}}}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            records.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(reply["status"]); self.end_headers()
            raw = reply["body"] if isinstance(reply["body"], bytes) else json.dumps(reply["body"]).encode()
            self.wfile.write(raw)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True); worker.start()
    config = {"service_url": f"http://127.0.0.1:{httpd.server_port}/virtual", "backend_kind": "loopback_contract_only",
              "backend_revision": "test-fixture", "allow_empty_toolbench_key": True, "timeout_seconds": 3}
    yield config, records, reply
    httpd.shutdown(); httpd.server_close(); worker.join()


def test_http_typed_payload_actual_return_and_no_gold(server, bindings):
    config, records, _ = server
    execute = create_executor(query_id="one", config=config, tool_bindings=bindings)
    arguments = {"text": "Paris", "nested": {"enabled": False, "count": 3, "items": [1, "2"]}}
    result = execute("city/search", arguments)
    assert json.loads(records[0]["tool_input"]) == arguments
    assert set(records[0]) == {"category", "tool_name", "api_name", "tool_input", "strip", "toolbench_key"}
    assert result.content == {"error": "", "response": {"city_id": 917}}
    assert result.success is True and result.metadata['transport_status'] == 'ok'
    assert result.metadata['backend_kind'] == 'loopback_contract_only'
    assert 'toolbench_key' not in result.metadata


@pytest.mark.parametrize('case', ['tool_error', 'http_error', 'not_json', 'missing_envelope', 'oversize'])
def test_http_failures_are_not_success(case, server, bindings):
    config, records, reply = server
    if case == 'tool_error': reply['body'] = {'error':'Unauthorized error...', 'response':''}
    if case == 'http_error': reply['status'] = 503
    if case == 'not_json': reply['body'] = b'<html>not a response</html>'
    if case == 'missing_envelope': reply['body'] = {'answer':'invented'}
    if case == 'oversize': config['max_response_bytes'] = 2
    result = create_executor(query_id='one', config=config, tool_bindings=bindings)('city/search', {'text':'Paris'})
    assert not result.success and result.content['error']
    assert len(records) == 1  # No hidden retries or gold fallback.


def test_binding_collision_unknown_identity_and_missing_key(server, bindings, monkeypatch):
    wrong = copy.deepcopy(bindings)
    wrong['other/id'] = {**bindings['city/search'], 'api_name': 'SEARCH'}
    with pytest.raises(ValueError, match='collide'): wire_bindings(wrong)
    config, records, _ = server
    execute = create_executor(query_id='one', config=config, tool_bindings=bindings)
    with pytest.raises(ValueError): execute('not/a/tool', {})
    assert records == []
    config.update(allow_empty_toolbench_key=False, toolbench_key_env='NATIVE_TEST_MISSING_KEY')
    monkeypatch.delenv('NATIVE_TEST_MISSING_KEY', raising=False)
    with pytest.raises(ValueError, match='credential'): create_executor(query_id='one', config=config, tool_bindings=bindings)


def test_loop_continuation_receipt_and_canonical_finish(server, tools, bindings):
    config, _, _ = server
    class Policy:
        def decide(self, history, registry):
            if len(history) == 1: return Decision('city/search', {'text':'Paris'}, 'Search.', ())
            assert history[-1]['content']['response']['city_id'] == 917
            return Decision(FINISH, {'return_type':'give_answer', 'final_answer':'917'}, '', ())
    trace = run_serial_agent(Policy(), 'Paris?', tools,
        create_executor(query_id='one',config=config,tool_bindings=bindings))
    assert trace['execution_receipts'][0]['transport_status'] == 'ok'
    assert trace['task_success_judged'] is False
    converted = convert_trace('Paris?', trace, tools, bindings)
    node = converted['answer']['answer_details'][0]
    while node['next']: node = node['next'][0]
    assert node['message']['name'] == 'Finish'
    assert json.loads(node['message']['arguments'])['final_answer'] == '917'


def test_raw_generation_failure_is_retained_without_fabricating_finish(tools, bindings):
    class Policy:
        def decide(self, *args):
            raise GenerationFailure('bad JSON',stage='arguments',text='{"oops":',identity='city/search',thought='Search.')
    trace=run_serial_agent(Policy(),'Paris?',tools,lambda *args: pytest.fail('must not execute invalid JSON'))
    assert trace['generation_failure']['generated_text'] == '{"oops":'
    converted=convert_trace('Paris?',trace,tools,bindings)
    assert converted['answer']['final_answer'] == ''
    assert converted['answer']['total_steps'] == 2


def test_converter_rejects_mismatched_actual_observation(tools, bindings):
    history=[{'type':'user','content':'Paris?'},*call_history('','city/search',{'text':'Paris'}),
             observation('weather/forecast', {'city_id':917})]
    with pytest.raises(ValueError,match='different API'):
        convert_trace('Paris?',{'history':history,'status':'call_budget_exhausted'},tools,bindings)


def test_format_matches_pinned_upstream_converter_and_get_steps(tools, bindings):
    root = os.environ.get('NATIVE_TEST_STABLETOOLBENCH_SOURCE')
    if not root: pytest.skip('Set an explicitly pinned upstream source for integration verification')
    root = Path(root) / 'toolbench/tooleval'
    # Load the original graph implementation without starting a judge/client.
    spec=importlib.util.spec_from_file_location('upstream_graph',root/'evaluation/dataclass.py')
    graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
    scope={'ExecutionGraph':graph.ExecutionGraph,'ExecutionNode':graph.ExecutionNode}
    wanted={'generate_init_message_node','process_valid_data'}
    tree=ast.parse((root/'convert_to_answer_format.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(root/'convert_to_answer_format.py'),'exec'),scope)
    tree=ast.parse((root/'utils.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_steps')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(root/'utils.py'),'exec'),scope)
    answer={'return_type':'give_answer','final_answer':'917'}
    history=[{'type':'user','content':'Paris?'},*call_history('Search.','city/search',{'text':'Paris'}),
             observation('city/search',{'error':'','response':{'city_id':917}}),*call_history('',FINISH,answer)]
    our=convert_trace('Paris?',{'history':history,'status':'give_answer'},tools,bindings)
    names=evaluator_names(tools,bindings)
    conversation=[{'role':'system','content':''},{'role':'user','content':'Paris?'},
        {'role':'assistant','content':'Search.'},
        {'role':'assistant','function_call':{'name':names['city/search'],'arguments':'{"text":"Paris"}'}},
        {'role':'function','content':'{"error":"","response":{"city_id":917}}'},
        {'role':'assistant','function_call':{'name':'Finish','arguments':json.dumps(answer,separators=(',',':'))}}]
    upstream=scope['process_valid_data']('NativeMemory.Serial',{'query':'Paris?','function':our['available_tools'],
        'final_answer':'917','train_messages':[conversation]})
    assert our == upstream
    steps, final=scope['get_steps'](our)
    assert "'name': 'Finish'" in final and names['city/search'] in steps
