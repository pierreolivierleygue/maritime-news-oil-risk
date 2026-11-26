// Editable PPTX fallback after the Spaces artifact service failed before authoring.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.ARTIFACT_RUNTIME_ROOT || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill=process.env.PRESENTATIONS_SKILL_DIR || path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1119.11507/skills/presentations');
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(process.env.RUNTIME_NODE_MODULES,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont();
const data=JSON.parse(await fs.readFile(path.join(root,'output/research/exhibit_data.json'),'utf8'));
const content=JSON.parse(await fs.readFile(path.join(root,'config/presentation_content_v1.json'),'utf8'));
const build=path.join(root,'tmp/reporting/presentation');
const output=path.join(root,'output/presentation');
await fs.mkdir(build,{recursive:true});await fs.mkdir(output,{recursive:true});
const finalPath=path.join(output,'maritime_research.pptx');
try { await fs.access(finalPath); throw new Error('Presentation exists; preserve it and use a new output revision.'); } catch(e) { if(e.code!=='ENOENT')throw e; }
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:'#142F43',teal:'#087F8C',gold:'#B97813',gray:'#8795A1',muted:'#526776',pale:'#E8F2F3',white:'#FFFFFF'};
function text(s,name,value,x,y,w,h,size=24,color=C.navy,bold=false){
  const a=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=value;a.text.style={typeface:font,fontSize:size,color,bold};return a;
}
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function slide(i){const s=p.slides.add();s.background.fill=C.white;
  rect(s,64,51,64,5,C.teal);
  text(s,'title',content.slides[i].title,64,72,1150,86,39,C.navy,true);
  text(s,'footer',`${i+1} / 6    Maritime news and oil-market uncertainty`,64,670,1130,24,15,C.muted);
  s.speakerNotes.text=`Suggested delivery: ${content.slides[i].seconds} seconds\n\n${content.slides[i].notes}\n\nSources:\n${content.slides[i].sources.join('\n')}`;
  return s;
}
function chart(s,type,config){
  // Embedded Excel chart workbooks use 12 significant digits; source results stay untouched.
  config={...config,series:config.series.map(series=>({...series,values:series.values.map(v=>v===null?null:Number(v.toPrecision(12)))}))};
  const c=s.charts.add(type,{
  chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
  titlePlacement:'none',hasLegend:true,
  legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:18,fill:C.navy}},
  xAxis:{textStyle:{typeface:font,fontSize:17,fill:C.navy},majorGridlines:null},
  yAxis:{textStyle:{typeface:font,fontSize:16,fill:C.muted},majorGridlines:{fill:'#E3E9EE',width:1},min:0},
  ...config});applyPresentationChartFont(c,{fontFamily:font});return c;
}
// 1. Question and hypothesis.
{
const s=slide(0);s.shapes.items.find(x=>x.name==='title').delete();
text(s,'cover-title','Can maritime news improve\noil-risk forecasts?',64,119,1150,170,56,C.navy,true);
text(s,'hypothesis','Hypothesis: threats and actual disruptions add information\nbeyond past market conditions and the amount of news.',68,346,1110,100,27);
text(s,'scope','Hormuz and the Red Sea / Bab el-Mandeb corridor\nRetrospective study, 2019-2025',68,504,1100,80,24,C.muted);
text(s,'date','Research presentation  /  24 November 2026',68,614,1110,34,19,C.muted);
}
// 2. Editable quarterly coverage and event-share charts.
{
const s=slide(1),r=data.coverage,categories=r.map(x=>x.quarter.slice(2).replace('-Q','.'));
text(s,'count-label','Observed exact texts per quarter',64,164,780,30,21,C.navy,true);
chart(s,'bar',{position:{left:64,top:201,width:790,height:205},categories,
 series:[{name:'Financial Times',values:r.map(x=>x.ft),fill:C.gray},{name:'Guardian',values:r.map(x=>x.guardian),fill:C.teal}],
 barOptions:{direction:'column',grouping:'stacked',gapWidth:45},xAxis:{visible:false},yAxis:{min:0,numberFormatCode:'0',textStyle:{fontSize:16,typeface:font},majorGridlines:{fill:'#E3E9EE',width:1}}});
text(s,'share-label','Event counts / all observed candidate texts',64,421,790,30,20,C.navy,true);
chart(s,'line',{position:{left:64,top:458,width:790,height:181},categories,lineOptions:{smooth:false},
 xAxis:{textStyle:{typeface:font,fontSize:10,fill:C.navy},majorGridlines:null},
 series:[{name:'Threat',values:r.map(x=>x.threat_share),line:{fill:C.teal,width:3},marker:{symbol:'none'}},
         {name:'Disruption',values:r.map(x=>x.disruption_share),line:{fill:C.gold,width:3},marker:{symbol:'none'}}],
 yAxis:{min:0,max:1,majorUnit:.5,numberFormatCode:'0%',textStyle:{fontSize:16,typeface:font},majorGridlines:{fill:'#E3E9EE',width:1}}});
text(s,'corpus',data.documents.toLocaleString('en-US'),903,185,300,62,49,C.teal,true);
text(s,'corpus-detail','classified articles\n858 FT + 691 Guardian',905,257,300,80,23);
text(s,'timing','End-of-Sunday cutoff\nNext-week observed OVX',905,365,305,82,22,C.navy,true);
text(s,'limits','FT gaps: 2019, 2020, 2024\nHistorical access unverified\n\nQuarterly display only.\nModels use weekly features.',905,477,305,165,20,C.muted);
}
// 3. NLP agreement, with limitations beside the evidence.
{
const s=slide(2),methods=['keyword','tfidf','embedding'];
chart(s,'bar',{position:{left:64,top:205,width:790,height:375},categories:['Relevance','Threat','Disruption'],
 series:methods.map((m,i)=>({name:['Keyword','TF-IDF','Embedding'][i],values:['relevant','threat','disruption'].map(t=>data.nlp[m].overall.targets[t].f1),fill:[C.teal,C.gray,C.gold][i]})),
 barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
 yAxis:{min:0,max:1.1,majorUnit:.25,numberFormatCode:'0.00',title:{text:'Positive-class F1',textStyle:{fontSize:18,typeface:font}},textStyle:{fontSize:16,typeface:font},majorGridlines:{fill:'#E3E9EE',width:1}},
 dataLabels:{showValue:true,numberFormatCode:'0.000',position:'outEnd',textStyle:{typeface:font,fontSize:17,fill:C.navy}}});
text(s,'holdout','120 reserved articles\nAgreement with AI references',66,597,790,55,20,C.muted);
text(s,'mean','0.803',907,205,290,63,48,C.teal,true);
text(s,'mean-label','Keyword mean F1',909,274,290,40,23);
text(s,'fragility','0.584',907,360,290,63,44,C.gold,true);
text(s,'fragility-label','without the largest\nrelated-report component',909,427,302,68,22);
text(s,'misses','All methods miss the three\nremaining disruption positives.',909,547,302,76,21,C.muted);
}
// 4. The economic equation, model nesting and chronological fit.
{
const s=slide(3);
text(s,'equation','Y(t+1) = a + bY(t) + cMean4(t) + dV(t) + eT(t) + fD(t)',68,170,1135,52,28,C.teal,true);
const values=[['Model','Inputs','Purpose'],['M0','Latest available weekly OVX','Persistence benchmark'],['M1','Latest OVX + trailing four-week mean','Market-history comparator'],['M2','M1 + log(1 + relevant text count)','News-quantity contribution'],['M3','M2 + threat share + disruption share','Primary comparison: M3 vs M2']];
const t=s.tables.add({rows:5,columns:3,left:68,top:249,width:1135,height:252,columnWidths:[100,607,428],values});
t.cells.block({row:0,column:0,rowCount:5,columnCount:3}).assign({fill:C.white,textStyle:{typeface:font,fontSize:21,color:C.navy},margins:{left:12,right:12,top:10,bottom:10}});
t.cells.block({row:0,column:0,rowCount:1,columnCount:3}).assign({fill:C.pale,textStyle:{bold:true}});
t.borders.assign({fill:'#D4DCE2',width:1,style:'solid'});
text(s,'split','2019-2022 training     2023 validation     2024-2025 final test',68,536,1135,42,24,C.navy,true);
text(s,'fit','Expanding fits use available outcomes only. Training-only scaling.\nFixed ridge penalty alpha = 0.01. Conservative timing also delays training labels.',68,590,1135,69,21,C.muted);
}
// 5. Native MAE comparison and primary interval.
{
const s=slide(4),m=data.finance.primary.models;
chart(s,'bar',{position:{left:68,top:199,width:745,height:360},categories:['M0','M1','M2','M3'],hasLegend:false,
 series:[{name:'MAE',values:['M0','M1','M2','M3'].map(k=>m[k].mae),fill:C.teal,points:[{idx:0,fill:C.gray},{idx:1,fill:'#57748A'},{idx:2,fill:'#70B8BE'},{idx:3,fill:C.teal}]}],
 barOptions:{direction:'column',grouping:'clustered',gapWidth:95},
 yAxis:{min:0,max:3.5,majorUnit:1,title:{text:'MAE, OVX points',textStyle:{fontSize:19,typeface:font}},textStyle:{fontSize:17,typeface:font},majorGridlines:{fill:'#E3E9EE',width:1}},
 dataLabels:{showValue:true,numberFormatCode:'0.0000',position:'outEnd',textStyle:{typeface:font,fontSize:21,fill:C.navy}}});
text(s,'key','M0 persistence   M1 market history\nM2 adds volume   M3 adds event shares',72,579,740,64,21,C.muted);
text(s,'gain','2.2%',873,199,334,68,52,C.teal,true);
text(s,'gain-label','lower MAE for M3 vs M2\n104 common test weeks',878,277,325,78,24);
text(s,'interval','M3 minus M2: -0.0608\n95% interval [-0.1490, +0.0227]',878,388,330,88,24,C.navy,true);
text(s,'interpretation','The interval includes zero.\nM2 and M3 RMSE is\nalmost identical.',878,517,330,108,23,C.muted);
}
// 6. Explicit limits and a genuinely new evaluation.
{
const s=slide(5);
text(s,'conclusion','A modest MAE gain does not establish a reliable forecasting advantage.',68,169,1125,76,28,C.teal,true);
text(s,'measurement-title','Measurement',68,285,510,39,27,C.navy,true);
text(s,'measurement','AI references and weak disruption recall\nlimit interpretation of event categories.',68,330,515,74,23,C.muted);
text(s,'timing-title','Timing and generalisation',68,444,520,39,27,C.navy,true);
text(s,'timing','Archive gaps and unverified availability.\nTwo test years and few crisis episodes.\nSource/timing checks give mixed support.',68,489,540,116,23,C.muted);
rect(s,649,284,3,320,'#D4DCE2');
text(s,'next-title','Next experiment',705,285,490,44,29,C.navy,true);
text(s,'next','Independently review more disruptions.\nFreeze a revised measurement method.\nEvaluate on genuinely new observations.',705,350,495,145,25);
text(s,'exposed','2024-2025 is now exposed.\nNo causal or trading claim.',705,538,495,77,24,C.gold,true);
}
if(p.slides.items.length!==6 || content.slides.reduce((a,s)=>a+s.seconds,0)!==420)throw new Error('Wrong slide count or timing');
// Render all six slides for visual QA before finalizing the editable package.
for(let i=0;i<6;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1.25});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await p.slides.items[i].export({format:'layout'});
 await fs.writeFile(path.join(build,`slide-${i+1}.layout.json`),new Uint8Array(await layout.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
 explicitTotalSlideCount:6,requiredNativeChartOwnerSlides:[2,3,5],requiredNativeTableOwnerSlides:[4],
 materializeLiteralChartWorkbooks:true,
 pythonExecutable:path.join(runtime,'python/bin/python'),
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','4'],
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation.json')});
const notes='# Speaker notes\n\nSix slides, seven minutes. Timing totals 420 seconds; rehearse rather than reading the sources aloud.\n\n'+content.slides.map((s,i)=>`## ${i+1}. ${s.title} (${s.seconds} seconds)\n\n${s.notes}\n\nSources: ${s.sources.join('; ')}\n`).join('\n');
await fs.writeFile(path.join(output,'speaker_notes.md'),notes);
const hash=async f=>createHash('sha256').update(await fs.readFile(f)).digest('hex');
await fs.writeFile(path.join(output,'manifest.json'),JSON.stringify({slides:6,minutes:7,font,
 authoring:'Bundled @oai/artifact-tool; editable native charts and table',
 fallback_reason:'Spaces artifact_preparation_failed, no content saved to blank cloud deck',
 source_sha256:{'output/research/exhibit_data.json':await hash(path.join(root,'output/research/exhibit_data.json')),
 'config/presentation_content_v1.json':await hash(path.join(root,'config/presentation_content_v1.json')),
 'scripts/build_presentation.mjs':await hash(fileURLToPath(import.meta.url))},
 artifact_sha256:{'maritime_research.pptx':await hash(finalPath),'speaker_notes.md':await hash(path.join(output,'speaker_notes.md'))}},null,2)+'\n');
console.log(JSON.stringify({finalPath,slides:6,font,validation:result},null,2));
