import json
from types import SimpleNamespace
import pytest
from test_toolbench_agent import tools
from latent_register.toolbench_agent import Decision, ExecutionResult
from latent_register.toolbench_data import FINISH
from latent_register.run_toolbench_e2e import load_queries, run_panel


def test_end_to_end_uses_executor_feedback_and_full_generation_budgets(tools, tmp_path):
    returned = {"city_id": "actually-returned-917"}
    class Policy:
        registration_profiles={}
        def register_tools(self, registry, **kwargs): pass
        def _clock(self, synchronized): return 0.0
        def decide(self, history, registry, **kwargs):
            assert kwargs == {"max_thought_tokens":1024,"max_argument_tokens":1024}
            if len(history)==1: return Decision("city/search",{"text":"Paris"},"",())
            assert history[-1]["content"]==returned
            return Decision(FINISH,{"return_type":"give_answer","final_answer":returned['city_id']},"",())
    factories=[]
    def factory(**kwargs):
        assert set(kwargs)=={'query_id','config','tool_bindings'}
        factories.append(kwargs['query_id'])
        def execute(identity, args):
            assert identity=='city/search' and args=={'text':'Paris'}
            return ExecutionResult(returned,True)
        return execute
    report=run_panel(Policy(),tools,[{'id':'one','query':'Paris?','split':'dev'}],factory,{}, {},tmp_path/'run',
                     thought_tokens=1024, argument_tokens=1024)
    assert report['episodes']==1 and report['task_success_judged'] is False
    assert factories==['one']
    trace=json.loads((tmp_path/'run/episodes.jsonl').read_text())['trace']
    assert trace['result']['final_answer']=='actually-returned-917'
    assert trace['costs']['executions_succeeded']==1


def test_query_panel_rejects_gold_fields_and_non_development_split(tmp_path):
    p=tmp_path/'queries.jsonl'
    p.write_text(json.dumps({'id':'x','query':'test','split':'dev','answer':'gold'})+'\n')
    with pytest.raises(ValueError): load_queries(p,'dev')
    p.write_text(json.dumps({'id':'x','query':'test','split':'test'})+'\n')
    with pytest.raises(ValueError): load_queries(p,'dev')
