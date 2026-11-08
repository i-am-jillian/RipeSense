const $=id=>document.getElementById(id);
let state=null, qrOpen=false, busy=false, qrId=null;
const steps=[['Inspecting the peel','Checking color and surface texture…'],['Reading its color spectrum','Simulating the chlorophyll profile…'],['Checking the ripening pattern','Simulating a ripening-gas reading…'],['Bringing it all together','Preparing your banana’s ripeness estimate…']];
function notice(text){$('notice').textContent=text;$('notice').hidden=!text;}
function render(s){
 state=s;const view=qrOpen&&s.status==='result'?'qr':s.status;
 ['idle','scanning','result','qr'].forEach(v=>$(v).hidden=v!==view);
 $('connection').textContent='● Connected';
 if(s.status==='scanning'){
  const identifying=s.progress<.125, detected=s.progress>=.125;
  const n=s.progress<.25?-1:Math.min(3,Math.floor((s.progress-.25)/.75*4));
  $('percent').textContent=Math.round(s.progress*100)+'%';$('progress').style.width=s.progress*100+'%';
  $('scan-heading').textContent=identifying?'IDENTIFYING YOUR FRUIT':s.progress<.25?'FRUIT IDENTIFIED':'SCANNING YOUR BANANA';
  $('detected-tag').textContent=detected?'✓ Banana detected':'Identifying…';
  document.querySelector('.chamber').classList.toggle('identified',detected);
  $('step-title').textContent=identifying?'What’s inside?':s.progress<.25?'Banana detected':steps[n][0];
  $('step-copy').textContent=identifying?'Looking at shape and peel color…':s.progress<.25?'Starting the ripeness scan…':steps[n][1];
  for(let i=0;i<4;i++)$('sensor'+i).className=i===n?'active':i<n?'done':'';
 }
 if(s.status==='result'){
  const ready=s.recipe_status==='ready';$('recipe').disabled=!ready;$('again').disabled=!ready;
  $('recipe').textContent=ready?'Get recipe · QR ↗':'Preparing recipe…';
  $('recipe-status').textContent=ready?(s.recipe_error?'Built-in recipe · Gemini unavailable':'Recipe created with Gemini'):(s.gemini_configured?'Gemini is writing your recipe…':'Preparing a built-in recipe…');
  if(ready){$('recipe-title').textContent=s.recipe.title;$('recipe-source').textContent=s.recipe.source;
   $('recipe-address').textContent=s.recipe_url;
   if(qrId!==s.scan_id){$('qr-image').src='/api/demo/qr/'+s.scan_id+'.svg';qrId=s.scan_id;}
  }
 }
}
async function scan(){if(busy)return;busy=true;$('start').disabled=true;notice('');try{const r=await fetch('/api/demo/scan',{method:'POST'});if(!r.ok)throw Error(r.status===403?'Start scans on the Pi touchscreen.':'Please wait for the current recipe.');qrOpen=false;await pollOnce();}catch(e){notice(e.message);}finally{busy=false;$('start').disabled=false;}}
async function pollOnce(){const r=await fetch('/api/demo/state');if(!r.ok)throw Error('Connection lost');render(await r.json());}
async function poll(){try{await pollOnce();}catch(e){$('connection').textContent='○ Reconnecting';}setTimeout(poll,500);}
async function newScan(){qrOpen=false;render({...state,status:'idle',scan_id:null,recipe:null,recipe_status:'idle',recipe_error:null});try{await fetch('/api/demo/reset',{method:'POST'});}catch(e){notice('Could not reset the scan. Please try again.');}}
$('start').onclick=scan;$('again').onclick=newScan;$('new-from-qr').onclick=newScan;$('recipe').onclick=()=>{qrOpen=true;render(state);};$('back').onclick=()=>{qrOpen=false;render(state);};poll();
