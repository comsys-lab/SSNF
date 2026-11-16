import unittest
from types import SimpleNamespace
from protocol import PrefillBarrier, page_capacity, validate_trial


class ProtocolTest(unittest.TestCase):
    def test_barrier_blocks_only_ready_requests_and_restores(self):
        a=SimpleNamespace(num_computed_tokens=128,num_prompt_tokens=128,num_output_tokens=1,next_decode_eligible_step=0)
        b=SimpleNamespace(num_computed_tokens=32,num_prompt_tokens=128,num_output_tokens=0,next_decode_eligible_step=4)
        scheduler=SimpleNamespace(requests={'a':a,'b':b})
        barrier=PrefillBarrier(scheduler,['a','b'])
        self.assertFalse(barrier.before_schedule())
        self.assertEqual(a.next_decode_eligible_step,10**15)
        self.assertEqual(b.next_decode_eligible_step,4)
        b.num_computed_tokens=128;b.num_output_tokens=1
        self.assertTrue(barrier.before_schedule())
        self.assertEqual(a.next_decode_eligible_step,0)
        self.assertEqual(b.next_decode_eligible_step,4)

    def test_barrier_waits_for_observed_first_sample_not_computed_only(self):
        a=SimpleNamespace(num_computed_tokens=128,num_prompt_tokens=128,num_output_tokens=0,next_decode_eligible_step=0)
        barrier=PrefillBarrier(SimpleNamespace(requests={'a':a}),['a'])
        self.assertFalse(barrier.before_schedule())
        a.num_output_tokens=1
        self.assertTrue(barrier.before_schedule())

    def test_paged_capacity_rounds_each_request_before_division(self):
        got=page_capacity(100,16,129,16)
        self.assertEqual(got['reserved_pages_per_request'],10)
        self.assertEqual(got['conservative_batch'],10)
        self.assertEqual(got['last_forward_pages_per_request'],9)

    def fixture(self):
        rows=[dict(request_id='a',finished=True,output_tokens=4),dict(request_id='b',finished=True,output_tokens=4)]
        steps=[dict(scheduled_ids=['a','b'],decode_ids=['a','b'],scheduled_tokens={'a':1,'b':1},
                    min_computed_before=128+i,input_tokens=128,wall_s=.1,preempted=[],barrier_released=True) for i in range(3)]
        return rows,steps

    def test_full_decode_and_last_length_required(self):
        rows,steps=self.fixture()
        got=validate_trial(rows,steps,2,4,'fixed')
        self.assertTrue(got['fixed_batch_valid'])
        self.assertTrue(got['full_batch_last_decode'])
        self.assertAlmostEqual(got['all_batch_decode_tok_s'],20)
        steps[-1]['decode_ids']=['a']
        got=validate_trial(rows,steps,2,4,'fixed')
        self.assertFalse(got['fixed_batch_valid'])
        self.assertFalse(got['full_batch_last_decode'])

    def test_completion_and_peak_do_not_prove_capacity(self):
        rows,steps=self.fixture()
        steps[-1]['min_computed_before']=129
        got=validate_trial(rows,steps,2,4,'capacity')
        self.assertTrue(got['complete_generation'])
        self.assertFalse(got['capacity_verified'])

    def test_preemption_disqualifies_capacity(self):
        rows,steps=self.fixture();steps[1]['preempted']=[{'request_id':'a'}]
        self.assertFalse(validate_trial(rows,steps,2,4,'capacity')['capacity_verified'])


if __name__=='__main__': unittest.main()
