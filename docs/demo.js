const drawings=[
  {name:"泵体法兰装配图",type:"装配图",ocr:"PUMP BODY / DN80",base:.94,r:18},
  {name:"离心泵壳体零件图",type:"零件图",ocr:"PUMP CASING",base:.91,r:42},
  {name:"法兰连接轴承座",type:"装配图",ocr:"BEARING / FLANGE",base:.88,r:7},
  {name:"阀体总成工程图",type:"装配图",ocr:"VALVE BODY / DN50",base:.85,r:25},
  {name:"传动轴与联轴器",type:"零件图",ocr:"SHAFT COUPLING",base:.82,r:50},
  {name:"支撑底座结构图",type:"结构图",ocr:"BASE SUPPORT",base:.79,r:3},
  {name:"管路法兰连接图",type:"装配图",ocr:"PIPE FLANGE",base:.77,r:35},
  {name:"齿轮箱壳体图",type:"零件图",ocr:"GEAR HOUSING",base:.74,r:10}
];
const $=id=>document.getElementById(id);let imageMode=false,lastQuery="";
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));b.classList.add("active");imageMode=b.dataset.mode==="image";$("textMode").classList.toggle("hidden",imageMode);$("imageMode").classList.toggle("hidden",!imageMode)});
$("topk").oninput=()=>{$("topkText").textContent=$("topk").value};$("ocrWeight").oninput=()=>{$("ocrText").textContent=$("ocrWeight").value+"%"};$("structureWeight").oninput=()=>{$("structureText").textContent=$("structureWeight").value+"%"};
$("imageInput").onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{$("preview").innerHTML=`<img src="${r.result}" alt="查询图预览">`};r.readAsDataURL(f)};
function card(d,i,boost){
  const clip=Math.min(.99,d.base+boost),structure=Math.min(.99,d.base-.04+Number($("structureWeight").value)/500),ocr=Math.min(.99,d.base-.12+Number($("ocrWeight").value)/400),score=clip*.48+structure*.32+ocr*.2;
  return `<article class="result"><div class="drawing"><span class="rank">TOP ${i+1}</span><div class="shape" style="--radius:${d.r}px;--w:${125+i*3}px"></div><div class="title-block"></div></div><div class="result-body"><h3>${d.name}</h3><div class="meta"><span>${d.type}</span><span>OCR: ${d.ocr}</span></div><div class="score"><b>${score.toFixed(4)}</b><small>CLIP ${clip.toFixed(2)} · 结构 ${structure.toFixed(2)} · OCR ${ocr.toFixed(2)}</small></div></div></article>`
}
function run(rerank=false){
  lastQuery=imageMode?"本地图纸查询":$("query").value.trim()||"通用结构图";const start=performance.now(),q=lastQuery.toLowerCase();let data=drawings.map((d,i)=>({...d,boost:(q.includes("泵")&&d.name.includes("泵")?.05:0)+(q.includes("法兰")&&d.name.includes("法兰")?.045:0)+(imageMode&&i<2?.035:0)}));data.sort((a,b)=>(b.base+b.boost)-(a.base+a.boost));const top=data.slice(0,Number($("topk").value));$("results").innerHTML=top.map((d,i)=>card(d,i,d.boost+(rerank?.012:0))).join("");$("summary").textContent=`“${lastQuery}” · 返回 ${top.length} 条结果 · 已完成 ${rerank?"结构/OCR 重排":"两阶段检索"}`;$("latency").textContent=(18+Math.round(performance.now()-start)+top.length*3)+" ms";
}
$("search").onclick=()=>run(false);$("imageSearch").onclick=()=>run(false);$("rerank").onclick=()=>run(true);run(false);
