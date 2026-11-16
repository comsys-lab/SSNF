import argparse,sys
from pathlib import Path
from engine import ppl,ppl_matched,serialization
from verified import VerifiedEngine
class TaskEngine(VerifiedEngine):
 def protocol(self,method):
  d,_=super().protocol(method);d.update(matched_extension_sha256=ppl.sha_file(Path(__file__)),serializer_sha256=ppl.sha_file(Path(serialization.__file__)),storage_layout='one MXFP4/6 per matrix or layer KV; block32 scale; map bits and tensor headers',total_budget_rule='W/K/V separate caps, 2K KV reference',frozen_map_protocol={'path':d['tensor_map_path'],'sha256':d['tensor_map_sha256']});return d,ppl.digest({k:v for k,v in d.items() if k!='hostname'})
p=argparse.ArgumentParser(add_help=False);p.add_argument('--selected-task',choices=['arc','gsm','longbench'],required=True);a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest];ppl_matched.MatchedEngine=TaskEngine
if a.selected_task=='arc':
 from tasks import multiple_choice
 multiple_choice.main()
elif a.selected_task=='gsm':
 from tasks import generate
 generate.main()
else:
 from tasks import adapters
 adapters.main()
