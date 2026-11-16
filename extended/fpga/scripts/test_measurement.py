import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_reference import seeded,advance,records,pack_states
from paired_analysis import jump,words,audit_drain,triplet_summary,analyze_window
from analyze import integrate

class MeasurementTests(unittest.TestCase):
    def test_jump_independent_sequential_and_composition(self):
        seed=seeded(910)
        x=seed
        for n in range(257):
            self.assertEqual(jump(seed,n),x);x=advance(x)
        self.assertEqual(jump(jump(seed,1234567),9876543),jump(seed,11111110))
    def test_long_audit_and_off_labels(self):
        c={'replicas':2,'compute':True,'format_id':0,'seed_set':3};n=8_000_000_123
        d={'status':2,'batches':n,'events':n*2,'accepted':n*2,'completed':n*2,'baseline_events':0,'source_states':[],'input_records':[],'results':[]}
        for r in range(2):
            last=jump(seeded(10003,r),n-1);rec,result,_=records(last)
            d['source_states'].append(words(pack_states(advance(last)),12));d['input_records'].append(words(rec[0],32));d['results'].append(result[0])
        self.assertTrue(audit_drain(c,d,{})['independent_jump_ahead_verified'])
        d['source_states'][1][5]^=1
        with self.assertRaisesRegex(ValueError,'source state'):audit_drain(c,d,{})
        d['source_states'][1][5]^=1;c['compute']=False
        with self.assertRaisesRegex(ValueError,'mislabeled'):audit_drain(c,d,{'initial_results':d['results']})
        d['completed']=0;d['baseline_events']=d['events']
        self.assertTrue(audit_drain(c,d,{'initial_results':d['results']})['last_result_or_off_hold_verified'])
    def test_integral_brackets_and_linear(self):
        self.assertEqual(integrate([(0,2),(1,4),(2,6)],.5,1.5),4)
        with self.assertRaisesRegex(ValueError,'bracket'):integrate([(0,2),(1,4)],-.1,.5)
    def test_triplet_retains_negative(self):
        base={'format_id':0,'replicas':16,'seed_set':0,'compute':False,'board_power_w':40,'fpga_temperature_min_C':40,'fpga_temperature_max_C':42}
        on={**base,'compute':True,'board_power_w':39,'dot_per_s':1.6e9,'board_nj_per_dot':24.375}
        result=triplet_summary([base,on,base]);self.assertEqual(result['activation_nj_per_dot'],-.625)
        with self.assertRaisesRegex(ValueError,'OFF/ON/OFF'):triplet_summary([on,on,base])
    def test_counter_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'config.json').write_text(json.dumps({'compute':False,'replicas':16,'interval':1}))
            a={'marker':'counter_start','status':1,'compute':False,'completed':1,'events':1,'baseline_events':1}
            (p/'window.jsonl').write_text(json.dumps(a)+'\n'+json.dumps({**a,'marker':'counter_end'}))
            with self.assertRaisesRegex(ValueError,'mislabeled'):analyze_window(p)
if __name__=='__main__':unittest.main()
