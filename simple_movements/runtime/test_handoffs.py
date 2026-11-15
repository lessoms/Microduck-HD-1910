"""Real shared-physics regression for queued walk->sit and turn->roll handoffs."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'runtime'))
from button_skills import ButtonController,DEFAULT_SIT,DEFAULT_ROLL

def main():
 (R/'outputs').mkdir(exist_ok=True)
 c=ButtonController(DEFAULT_SIT,DEFAULT_ROLL,20261111)
 def until(state,limit=1600):
  for _ in range(limit):
   c.step()
   if c.state=='fault':raise AssertionError(c.events[-1])
   if c.state==state and c.stable_ticks>=8 and c.idle_ticks>=50:return
  raise AssertionError(('timeout',state,c.state))
 try:
  until('standing');assert c.set_velocity(.15,0.)
  for _ in range(220):c.step();assert c.state!='fault'
  assert c.command('sit');until('sitting');assert c.command('stand');until('standing')
  assert c.set_velocity(0.,.5)
  for _ in range(220):c.step();assert c.state!='fault'
  assert c.command('roll');until('standing')
  for _ in range(100):c.step();assert c.state!='fault'
  assert c.reset_count==1
  assert sum(x['event']=='queued_after_stop' for x in c.events)==2
  report=dict(passed=True,state=c.state,reset_count=c.reset_count,elapsed_s=c.ticks*.02,events=c.events)
  (R/'outputs/packaged_handoffs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  print('Combined queued handoffs passed',flush=True)
 finally:c.close()
if __name__=='__main__':main()
