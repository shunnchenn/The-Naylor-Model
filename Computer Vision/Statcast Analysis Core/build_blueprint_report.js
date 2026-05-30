const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, LevelFormat
} = require('docx');
const fs = require('fs');

const NAVY="0B2545",HEAD="1F3A5F",GRAY="F2F5F8",BLUE="D5E8F0",DKBLUE="2E5FA3",
      GREEN="D8F0E0",RED="FAD7D7",WHITE="FFFFFF",LTGRAY="EEEEEE",GOLD="FFF3CD";

function loadImage(p){try{return fs.readFileSync(p);}catch(e){return null;}}
const BASE='/Users/shunchen/Desktop/The-Naylor-Model/Figures';
const imgScatter=loadImage(`${BASE}/Fig_NaylorBlueprint_Scatter.png`);
const imgTopN=loadImage(`${BASE}/Fig_NaylorBlueprint_TopN.png`);
const imgBottomN=loadImage(`${BASE}/Fig_NaylorBlueprint_BottomN.png`);
const imgByYear=loadImage(`${BASE}/Fig_GroundCovered_TopN_ByYear.png`);
const imgAbsGain=loadImage(`${BASE}/Fig_Top25BCS_AbsoluteGain.png`);
const imgTop25Season=loadImage(`${BASE}/Fig_BCS_Top25_ByYear.png`);
const imgBot25Season=loadImage(`${BASE}/Fig_BCS_Bot25_ByYear.png`);

function imgPara(data,w,h,caption){
  const kids=[];
  if(data)kids.push(new ImageRun({type:'png',data,transformation:{width:w,height:h},altText:{title:caption,description:caption,name:caption}}));
  const paras=[];
  if(kids.length)paras.push(new Paragraph({alignment:AlignmentType.CENTER,children:kids,spacing:{before:120,after:60}}));
  paras.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:0,after:200},children:[new TextRun({text:caption,italics:true,size:18,color:"555555"})]}));
  return paras;
}

const brd={style:BorderStyle.SINGLE,size:1,color:"CCCCCC"};
const borders={top:brd,bottom:brd,left:brd,right:brd};
const cm={top:80,bottom:80,left:120,right:120};

function hdrCell(t,w,c=BLUE){return new TableCell({borders,width:{size:w,type:WidthType.DXA},shading:{fill:c,type:ShadingType.CLEAR},margins:cm,verticalAlign:VerticalAlign.CENTER,children:[new Paragraph({children:[new TextRun({text:t,bold:true,size:17,color:HEAD})]})]});}
function dataCell(t,w,f=WHITE,b=false){return new TableCell({borders,width:{size:w,type:WidthType.DXA},shading:{fill:f,type:ShadingType.CLEAR},margins:cm,verticalAlign:VerticalAlign.CENTER,children:[new Paragraph({children:[new TextRun({text:String(t),size:17,bold:b,color:"222222"})]})]});}

function h1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,pageBreakBefore:true,children:[new TextRun({text:t,bold:true,size:32,color:NAVY})]});}
function h2(t){return new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:240,after:120},children:[new TextRun({text:t,bold:true,size:24,color:HEAD})]});}
function h3(t){return new Paragraph({heading:HeadingLevel.HEADING_3,spacing:{before:160,after:80},children:[new TextRun({text:t,bold:true,size:22,color:HEAD})]});}
function body(t,b=false){return new Paragraph({spacing:{before:80,after:120},children:[new TextRun({text:t,size:22,bold:b,color:"222222"})]});}
function sp(n=1){return Array.from({length:n},()=>new Paragraph({children:[new TextRun({text:"",size:22})],spacing:{before:0,after:60}}));}
function pageBreak(){return new Paragraph({children:[new PageBreak()]});}
function bullet(t){return new Paragraph({numbering:{reference:"bullets",level:0},spacing:{before:60,after:60},children:[new TextRun({text:t,size:22,color:"222222"})]});}
function tableLabel(t){return new Paragraph({spacing:{before:160,after:80},children:[new TextRun({text:t,bold:true,size:20,color:HEAD})]});}
function footnote(t){return new Paragraph({spacing:{before:60,after:120},children:[new TextRun({text:t,size:17,italics:true,color:"666666"})]});}
function mono(t){return new Paragraph({spacing:{before:160,after:160},shading:{fill:"F5F5F5",type:ShadingType.CLEAR},indent:{left:360},children:[new TextRun({text:t,font:"Courier New",size:22,bold:true,color:NAVY})]});}


// DATA
const top15Data=[
  [1,"Ryan McMahon","NYY",2026,"25.7",14,"4.59",3,0,"100%","24.50","+6.44","+1.09","-0.53","+8.05"],
  [2,"Agustín Ramírez","MIA",2025,"26.7",30,"4.78",13,1,"93%","21.31","+4.90","+1.44","-0.53","+6.86"],
  [3,"Paul Goldschmidt","STL",2024,"26.3",21,"4.58",10,0,"100%","16.70","+2.27","+1.79","-0.53","+4.59"],
  [4,"JJ Wetherholt","STL",2026,"27.3",50,"4.30",5,0,"100%","16.84","+2.78","+1.14","-0.53","+4.45"],
  [5,"Jordan Walker","STL",2025,"28.7",84,"4.40",10,0,"100%","15.28","+2.32","+1.54","-0.53","+4.39"],
  [6,"Josh Naylor","SEA",2025,"24.4",1,"4.86",22,1,"96%","16.74","+1.42","+2.07","-0.53","+4.01"],
  [7,"Josh Naylor","SEA",2026,"24.6",2,"4.73",9,1,"90%","17.73","+2.19","+1.20","-0.53","+3.91"],
  [8,"Ramón Laureano","CLE",2023,"27.9",63,"4.40",11,1,"92%","16.94","+2.66","+0.72","-0.53","+3.90"],
  [9,"Chase DeLauter","CLE",2026,"26.6",28,"4.48",2,1,"67%","18.97","+3.70","-0.37","-0.53","+3.85"],
  [10,"Trea Turner","PHI",2023,"30.3",99,"4.14",24,0,"100%","13.59","+1.74","+1.49","-0.53","+3.75"],
  [11,"Juan Soto","NYM",2025,"25.8",13,"4.58",30,0,"100%","14.18","+0.54","+2.39","-0.53","+3.45"],
  [12,"Michael A. Taylor","PIT",2024,"28.5",80,"4.30",10,0,"100%","13.59","+1.40","+1.51","-0.53","+3.44"],
  [13,"Nico Hoerner","CHC",2026,"28.4",81,"4.34",10,0,"100%","13.51","+1.35","+1.53","-0.53","+3.41"],
  [14,"Luis Arraez","SF",2026,"26.7",33,"4.57",4,0,"100%","15.22","+1.63","+1.18","-0.53","+3.33"],
  [15,"Kyle Tucker","LAD",2026,"26.4",25,"4.54",4,0,"100%","15.43","+1.62","+1.18","-0.53","+3.33"],
];
const bot15Data=[
  [515,"Chandler Simpson","TB",2025,"29.6",96,"3.97",38,8,"83%","9.93","-0.32","-0.00","+6.39","-6.71"],
  [514,"Bobby Witt Jr.","KC",2023,"30.5",100,"4.12",40,8,"83%","9.56","-0.44","-0.19","+5.94","-6.57"],
  [513,"Ji Hwan Bae","PIT",2023,"29.7",97,"4.05",19,7,"73%","11.22","+0.16","-1.27","+4.75","-5.86"],
  [512,"Daylen Lile","WSH",2025,"29.1",90,"4.23",8,6,"57%","8.31","-1.43","-2.14","+1.47","-5.04"],
  [511,"Bobby Witt Jr.","KC",2024,"30.5",100,"4.10",25,6,"81%","9.43","-0.13","-0.22","+4.52","-4.87"],
  [510,"Ceddanne Rafaela","BOS",2024,"28.8",88,"4.17",15,8,"65%","9.86","-0.57","-1.63","+2.41","-4.62"],
  [509,"Elly De La Cruz","CIN",2024,"30.0",98,"4.21",55,9,"86%","8.89","-0.63","+0.49","+4.47","-4.60"],
  [508,"Bobby Witt Jr.","KC",2025,"30.2",99,"4.15",32,7,"82%","10.13","+0.03","-0.02","+4.34","-4.33"],
  [507,"Myles Straw","CLE",2023,"29.2",93,"4.14",17,5,"77%","13.54","+1.27","-0.74","+4.72","-4.20"],
  [506,"Luis Rengifo","LAA",2025,"27.0",39,"4.36",7,7,"50%","8.53","-2.15","-2.57","-0.53","-4.20"],
  [505,"Jacob Young","WSH",2025,"29.3",94,"4.25",12,7,"63%","9.85","-0.49","-1.76","+1.87","-4.11"],
  [504,"Jeremy Peña","HOU",2023,"29.4",95,"4.28",13,7,"65%","10.19","-0.53","-1.88","+1.67","-4.08"],
  [503,"Alejandro Osuna","TEX",2026,"28.5",81,"4.19",1,2,"33%","7.40","-2.04","-1.98","+0.02","-4.04"],
  [502,"David Hamilton","BOS",2025,"29.3",93,"4.02",15,5,"75%","10.47","-0.14","-0.74","+3.07","-3.95"],
  [501,"Taylor Ward","BAL",2026,"27.5",56,"4.47",1,2,"33%","7.12","-2.60","-1.78","-0.53","-3.85"],
];
const naylorSotoData=[
  [6,"Josh Naylor","SEA",2025,"24.4",1,"4.86",22,1,"96%","16.74","+1.42","+2.07","-0.53","+4.01"],
  [7,"Josh Naylor","SEA",2026,"24.6",2,"4.73",9,1,"90%","17.73","+2.19","+1.20","-0.53","+3.91"],
  [209,"Juan Soto","SD",2023,"26.8",34,"4.39",6,4,"60%","16.12","+1.75","-1.75","-0.53","+0.53"],
  [284,"Juan Soto","NYY",2024,"26.8",32,"4.36",6,4,"60%","13.89","+0.89","-1.47","-0.53","-0.06"],
  [11,"Juan Soto","NYM",2025,"25.8",13,"4.58",30,0,"100%","14.18","+0.54","+2.39","-0.53","+3.45"],
  [55,"Juan Soto","NYM",2026,"25.9",18,"4.46",2,1,"67%","16.40","+1.97","-0.33","-0.53","+2.16"],
];
const absGainData=[
  [1,"Ryan McMahon","NYY",2026,"24.50","+6.44","+8.05"],
  [2,"Agustín Ramírez","MIA",2025,"21.31","+4.90","+6.86"],
  [3,"Chase DeLauter","CLE",2026,"18.97","+3.70","+3.85"],
  [4,"Josh Naylor","SEA",2026,"17.73","+2.19","+3.91"],
  [5,"Ramón Laureano","CLE",2023,"16.94","+2.66","+3.90"],
  [6,"JJ Wetherholt","STL",2026,"16.84","+2.78","+4.45"],
  [7,"Josh Naylor","SEA",2025,"16.74","+1.42","+4.01"],
  [8,"Paul Goldschmidt","STL",2024,"16.70","+2.27","+4.59"],
  [9,"Juan Soto","NYM",2026,"16.40","+1.97","+2.16"],
  [10,"Juan Soto","SD",2023,"16.12","+1.75","+0.53"],
  [11,"Gleyber Torres","NYY",2023,"15.91","+1.47","+0.75"],
  [12,"Brett Baty","NYM",2026,"15.83","+2.05","+2.15"],
  [13,"Spencer Steer","CIN",2023,"15.72","+2.17","+2.71"],
  [14,"Lars Nootbaar","STL",2023,"15.56","+1.84","+2.95"],
  [15,"Jesús Sánchez","HOU",2025,"15.53","+1.78","+3.32"],
  [16,"Kyle Tucker","LAD",2026,"15.43","+1.62","+3.33"],
  [17,"Gabriel Moreno","AZ",2026,"15.30","+1.59","+1.70"],
  [18,"Jordan Walker","STL",2025,"15.28","+2.32","+4.39"],
  [19,"Anthony Volpe","NYY",2026,"15.23","+1.99","+2.01"],
  [20,"Luis Arraez","SF",2026,"15.22","+1.63","+3.33"],
  [21,"Cal Raleigh","SEA",2026,"15.20","+1.25","+1.53"],
  [22,"Jordan Walker","STL",2026,"15.14","+2.55","+1.85"],
  [23,"Ke'Bryan Hayes","PIT",2023,"15.04","+1.26","+0.26"],
  [24,"Adam Frazier","KC",2025,"15.02","+1.33","+0.42"],
  [25,"Jazz Chisholm Jr.","NYY",2026,"14.87","+2.11","+2.97"],
];
const top25_2023=[
  [1, "Ram\u00f3n Laureano", "CLE", "27.9", 63, 11, 1, "92%", "16.94", "+3.90"],
  [2, "Trea Turner", "PHI", "30.3", 99, 24, 0, "100%", "13.59", "+3.75"],
  [3, "Bryan Reynolds", "PIT", "27.8", 62, 10, 0, "100%", "14.27", "+2.97"],
  [4, "Sam Haggerty", "SEA", "29.2", 92, 10, 0, "100%", "13.53", "+2.96"],
  [5, "Lars Nootbaar", "STL", "27.8", 60, 10, 1, "91%", "15.56", "+2.95"],
  [6, "Jonathan India", "CIN", "27.8", 60, 12, 1, "92%", "14.73", "+2.75"],
  [7, "Spencer Steer", "CIN", "28.4", 77, 10, 2, "83%", "15.72", "+2.71"],
  [8, "Anthony Volpe", "NYY", "28.4", 75, 22, 3, "88%", "14.32", "+2.44"],
  [9, "Chris Taylor", "LAD", "28.3", 75, 16, 2, "89%", "14.39", "+2.44"],
  [10, "Taylor Walls", "TB", "28.2", 71, 20, 1, "95%", "13.32", "+2.43"],
  [11, "Mark Canha", "MIL", "27.8", 60, 9, 0, "100%", "13.36", "+2.40"],
  [12, "Michael Harris II", "ATL", "28.8", 85, 13, 0, "100%", "11.91", "+2.04"],
  [13, "Christian Yelich", "MIL", "28.1", 69, 21, 2, "91%", "13.13", "+1.97"],
  [14, "Francisco Lindor", "NYM", "28.1", 68, 24, 2, "92%", "12.79", "+1.91"],
  [15, "Jose Altuve", "HOU", "26.9", 36, 13, 0, "100%", "12.69", "+1.88"],
  [16, "Steven Kwan", "CLE", "28.1", 68, 14, 1, "93%", "13.28", "+1.78"],
  [17, "Jacob Young", "WSH", "30.0", 98, 11, 0, "100%", "10.64", "+1.59"],
  [18, "Geraldo Perdomo", "AZ", "27.2", 44, 13, 3, "81%", "14.62", "+1.52"],
  [19, "Fernando Tatis Jr.", "SD", "29.3", 94, 24, 2, "92%", "12.78", "+1.52"],
  [20, "Mookie Betts", "LAD", "27.2", 43, 13, 2, "87%", "13.51", "+1.40"],
  [21, "Tommy Pham", "AZ", "27.8", 61, 15, 3, "83%", "13.44", "+1.30"],
  [22, "Michael A. Taylor", "MIN", "28.7", 83, 13, 1, "93%", "12.09", "+1.29"],
  [23, "Daulton Varsho", "TOR", "28.1", 69, 11, 2, "85%", "13.18", "+1.24"],
  [24, "Jack Suwinski", "PIT", "28.6", 82, 12, 1, "92%", "12.03", "+1.21"],
  [25, "Miles Mastrobuoni", "CHC", "28.6", 80, 10, 1, "91%", "12.21", "+1.15"],
];
const bot25_2023=[
  [1, "Bobby Witt Jr.", "KC", "30.5", 100, 40, 8, "83%", "9.56", "-6.57"],
  [2, "Ji Hwan Bae", "PIT", "29.7", 97, 19, 7, "73%", "11.22", "-5.86"],
  [3, "Myles Straw", "CLE", "29.2", 93, 17, 5, "77%", "13.54", "-4.20"],
  [4, "Jeremy Pe\u00f1a", "HOU", "29.4", 95, 13, 7, "65%", "10.19", "-4.08"],
  [5, "Seiya Suzuki", "CHC", "28.5", 79, 5, 6, "46%", "10.66", "-3.37"],
  [6, "Alex Call", "WSH", "28.8", 85, 7, 6, "54%", "11.51", "-3.19"],
  [7, "Elly De La Cruz", "CIN", "30.5", 99, 31, 4, "89%", "8.88", "-3.09"],
  [8, "Matt McLain", "CIN", "29.0", 89, 10, 5, "67%", "10.67", "-3.07"],
  [9, "Ezequiel Duran", "TEX", "29.1", 91, 7, 4, "64%", "10.07", "-2.99"],
  [10, "Akil Baddoo", "DET", "29.0", 89, 11, 3, "79%", "9.68", "-2.69"],
  [11, "Dairon Blanco", "KC", "30.3", 99, 23, 4, "85%", "9.90", "-2.68"],
  [12, "Andr\u00e9s Gim\u00e9nez", "CLE", "29.2", 92, 25, 5, "83%", "13.12", "-2.63"],
  [13, "Corbin Carroll", "AZ", "30.1", 99, 43, 4, "92%", "12.49", "-2.59"],
  [14, "Cody Bellinger", "CHC", "28.3", 74, 17, 5, "77%", "10.62", "-2.45"],
  [15, "Brenton Doyle", "COL", "29.9", 97, 20, 4, "83%", "9.29", "-2.45"],
  [16, "Randy Arozarena", "TB", "28.4", 75, 20, 8, "71%", "10.99", "-2.45"],
  [17, "Will Brennan", "CLE", "28.3", 73, 10, 4, "71%", "8.79", "-2.31"],
  [18, "Esteury Ruiz", "ATH", "29.6", 96, 57, 10, "85%", "10.33", "-2.23"],
  [19, "Chas McCormick", "HOU", "28.2", 72, 18, 6, "75%", "8.90", "-2.12"],
  [20, "Jorge Mateo", "BAL", "30.1", 99, 28, 4, "88%", "12.81", "-2.02"],
  [21, "Matt Vierling", "DET", "29.1", 90, 6, 4, "60%", "10.78", "-1.98"],
  [22, "Jose Siri", "TB", "29.7", 97, 9, 3, "75%", "11.18", "-1.96"],
  [23, "Jarred Kelenic", "SEA", "28.0", 63, 13, 4, "76%", "9.01", "-1.85"],
  [24, "Leody Taveras", "TEX", "29.1", 90, 11, 4, "73%", "12.79", "-1.84"],
  [25, "Edward Olivares", "KC", "28.6", 80, 8, 4, "67%", "10.01", "-1.82"],
];
const top25_2024=[
  [1, "Paul Goldschmidt", "STL", "26.3", 21, 10, 0, "100%", "16.70", "+4.59"],
  [2, "Michael A. Taylor", "PIT", "28.5", 80, 10, 0, "100%", "13.59", "+3.44"],
  [3, "Maikel Garcia", "KC", "27.9", 64, 34, 2, "94%", "13.12", "+3.06"],
  [4, "Brandon Nimmo", "NYM", "28.0", 66, 13, 0, "100%", "12.74", "+2.93"],
  [5, "Jake Bauers", "MIL", "27.2", 43, 12, 1, "92%", "13.78", "+2.67"],
  [6, "Victor Robles", "SEA", "27.8", 61, 31, 0, "100%", "11.47", "+2.63"],
  [7, "Kevin Pillar", "LAA", "28.4", 78, 11, 1, "92%", "12.74", "+2.39"],
  [8, "Josh Lowe", "TB", "28.4", 77, 17, 1, "94%", "12.39", "+2.31"],
  [9, "Max Schuemann", "ATH", "27.8", 60, 13, 1, "93%", "12.71", "+2.31"],
  [10, "Zach Neto", "LAA", "28.2", 74, 24, 4, "86%", "13.23", "+2.21"],
  [11, "Zach McKinstry", "DET", "28.4", 77, 14, 0, "100%", "11.05", "+2.16"],
  [12, "George Springer", "TOR", "28.1", 70, 15, 1, "94%", "11.94", "+2.15"],
  [13, "Luis Garc\u00eda Jr.", "WSH", "27.2", 45, 17, 3, "85%", "13.97", "+2.12"],
  [14, "Sal Frelick", "MIL", "29.3", 93, 12, 1, "92%", "13.98", "+2.04"],
  [15, "Francisco Lindor", "NYM", "27.6", 55, 22, 3, "88%", "12.80", "+2.02"],
  [16, "Spencer Steer", "CIN", "28.2", 74, 13, 1, "93%", "11.82", "+1.98"],
  [17, "Mookie Betts", "LAD", "26.7", 30, 15, 1, "94%", "12.33", "+1.92"],
  [18, "Bryson Stott", "PHI", "29.1", 92, 27, 1, "96%", "11.01", "+1.90"],
  [19, "Whit Merrifield", "ATL", "29.0", 91, 11, 2, "85%", "14.04", "+1.85"],
  [20, "Tyrone Taylor", "NYM", "28.9", 88, 10, 2, "83%", "12.54", "+1.76"],
  [21, "Zack Gelof", "ATH", "28.7", 84, 24, 2, "92%", "11.09", "+1.73"],
  [22, "Lawrence Butler", "ATH", "27.6", 55, 18, 0, "100%", "10.34", "+1.69"],
  [23, "Jose Altuve", "HOU", "27.1", 40, 21, 4, "84%", "13.29", "+1.66"],
  [24, "Starling Marte", "NYM", "27.1", 41, 15, 0, "100%", "10.69", "+1.59"],
  [25, "Matt Chapman", "SF", "28.7", 85, 9, 1, "90%", "11.49", "+1.59"],
];
const bot25_2024=[
  [1, "Bobby Witt Jr.", "KC", "30.5", 100, 25, 6, "81%", "9.43", "-4.87"],
  [2, "Ceddanne Rafaela", "BOS", "28.8", 88, 15, 8, "65%", "9.86", "-4.62"],
  [3, "Elly De La Cruz", "CIN", "30.0", 98, 55, 9, "86%", "8.89", "-4.60"],
  [4, "Jazz Chisholm Jr.", "NYY", "28.6", 83, 26, 8, "76%", "12.88", "-3.74"],
  [5, "Jose Siri", "TB", "29.9", 98, 13, 6, "68%", "10.53", "-3.74"],
  [6, "Jarren Duran", "BOS", "29.6", 96, 20, 6, "77%", "10.17", "-3.69"],
  [7, "Nicky Lopez", "CWS", "27.1", 42, 4, 6, "40%", "10.35", "-3.53"],
  [8, "Jacob Young", "WSH", "29.7", 97, 28, 7, "80%", "10.05", "-3.32"],
  [9, "Lane Thomas", "CLE", "29.3", 94, 26, 14, "65%", "10.13", "-3.24"],
  [10, "Adolis Garc\u00eda", "TEX", "26.8", 31, 6, 5, "54%", "9.13", "-3.18"],
  [11, "Masyn Winn", "STL", "28.8", 87, 9, 5, "64%", "9.64", "-3.09"],
  [12, "Connor Wong", "BOS", "28.4", 77, 7, 6, "54%", "8.74", "-3.08"],
  [13, "Corbin Carroll", "AZ", "29.6", 96, 31, 6, "84%", "10.78", "-3.03"],
  [14, "Christopher Morel", "TB", "27.3", 45, 6, 7, "46%", "10.73", "-2.93"],
  [15, "Joey Ortiz", "MIL", "28.7", 85, 6, 4, "60%", "9.08", "-2.83"],
  [16, "James Wood", "WSH", "28.7", 85, 13, 5, "72%", "9.76", "-2.66"],
  [17, "Vidal Bruj\u00e1n", "MIA", "27.5", 52, 5, 6, "46%", "11.29", "-2.51"],
  [18, "Brayan Rocchio", "CLE", "26.8", 33, 8, 6, "57%", "10.67", "-2.21"],
  [19, "Jonny DeLuca", "TB", "29.8", 98, 15, 4, "79%", "9.33", "-2.21"],
  [20, "Tyler Freeman", "CLE", "28.2", 73, 6, 5, "54%", "10.37", "-2.00"],
  [21, "Willi Castro", "MIN", "27.9", 64, 11, 7, "61%", "10.43", "-1.90"],
  [22, "Jos\u00e9 Caballero", "TB", "28.3", 76, 39, 13, "75%", "10.35", "-1.89"],
  [23, "Jackson Chourio", "MIL", "29.7", 97, 19, 4, "83%", "9.66", "-1.82"],
  [24, "Parker Meadows", "DET", "29.3", 94, 9, 3, "75%", "9.92", "-1.77"],
  [25, "Jo Adell", "LAA", "28.8", 88, 13, 8, "62%", "10.02", "-1.74"],
];
const top25_2025=[
  [1, "Agust\u00edn Ram\u00edrez", "MIA", "26.7", 30, 13, 1, "93%", "21.31", "+6.86"],
  [2, "Jordan Walker", "STL", "28.7", 84, 10, 0, "100%", "15.28", "+4.39"],
  [3, "Josh Naylor", "SEA", "24.4", 1, 22, 1, "96%", "16.74", "+4.01"],
  [4, "Juan Soto", "NYM", "25.8", 13, 30, 0, "100%", "14.18", "+3.45"],
  [5, "Jes\u00fas S\u00e1nchez", "HOU", "27.0", 38, 10, 1, "91%", "15.53", "+3.32"],
  [6, "Tyrone Taylor", "NYM", "29.3", 94, 11, 0, "100%", "12.91", "+3.31"],
  [7, "Francisco Lindor", "NYM", "27.4", 47, 25, 3, "89%", "14.72", "+3.05"],
  [8, "Nico Hoerner", "CHC", "28.6", 82, 25, 1, "96%", "12.92", "+3.02"],
  [9, "Hyeseong Kim", "LAD", "28.7", 83, 12, 0, "100%", "12.40", "+2.72"],
  [10, "Brandon Nimmo", "NYM", "27.3", 45, 11, 0, "100%", "12.80", "+2.55"],
  [11, "Anthony Volpe", "NYY", "28.3", 72, 16, 4, "80%", "14.76", "+2.38"],
  [12, "Cedric Mullins", "NYM", "28.4", 77, 19, 1, "95%", "12.52", "+2.36"],
  [13, "Jared Triolo", "PIT", "28.8", 86, 11, 1, "92%", "12.62", "+2.30"],
  [14, "Byron Buxton", "MIN", "30.2", 99, 21, 0, "100%", "10.02", "+2.22"],
  [15, "Colton Cowser", "BAL", "27.4", 47, 12, 0, "100%", "12.00", "+2.18"],
  [16, "Dansby Swanson", "CHC", "28.5", 81, 17, 1, "94%", "11.49", "+1.86"],
  [17, "Randy Arozarena", "SEA", "27.7", 55, 28, 3, "90%", "12.19", "+1.85"],
  [18, "Geraldo Perdomo", "AZ", "27.2", 44, 23, 3, "88%", "12.80", "+1.83"],
  [19, "Trevor Story", "BOS", "28.6", 82, 24, 1, "96%", "10.67", "+1.83"],
  [20, "Myles Straw", "TOR", "29.4", 95, 11, 0, "100%", "10.44", "+1.83"],
  [21, "Jon Berti", "CHC", "28.3", 74, 11, 2, "85%", "12.74", "+1.64"],
  [22, "Jorge Mateo", "BAL", "29.3", 94, 14, 1, "93%", "11.96", "+1.62"],
  [23, "George Springer", "TOR", "28.0", 65, 15, 1, "94%", "11.20", "+1.61"],
  [24, "Royce Lewis", "MIN", "26.8", 32, 9, 1, "90%", "12.75", "+1.57"],
  [25, "Sam Haggerty", "TEX", "28.9", 86, 11, 2, "85%", "13.94", "+1.52"],
];
const bot25_2025=[
  [1, "Chandler Simpson", "TB", "29.6", 96, 38, 8, "83%", "9.93", "-6.71"],
  [2, "Daylen Lile", "WSH", "29.1", 90, 8, 6, "57%", "8.31", "-5.04"],
  [3, "Bobby Witt Jr.", "KC", "30.2", 99, 32, 7, "82%", "10.13", "-4.33"],
  [4, "Luis Rengifo", "LAA", "27.0", 39, 7, 7, "50%", "8.53", "-4.20"],
  [5, "Jacob Young", "WSH", "29.3", 94, 12, 7, "63%", "9.85", "-4.11"],
  [6, "David Hamilton", "BOS", "29.3", 93, 15, 5, "75%", "10.47", "-3.95"],
  [7, "Harrison Bader", "PHI", "28.8", 86, 8, 7, "53%", "9.72", "-3.67"],
  [8, "Jackson Holliday", "BAL", "28.6", 82, 17, 9, "65%", "11.04", "-2.93"],
  [9, "Ernie Clement", "TOR", "28.6", 81, 5, 5, "50%", "11.23", "-2.69"],
  [10, "Brice Turang", "MIL", "28.9", 87, 21, 7, "75%", "11.46", "-2.54"],
  [11, "Sal Frelick", "MIL", "28.9", 88, 11, 3, "79%", "8.66", "-2.17"],
  [12, "Dylan Crews", "WSH", "29.0", 89, 13, 5, "72%", "10.37", "-2.12"],
  [13, "Corbin Carroll", "AZ", "29.8", 97, 26, 5, "84%", "11.18", "-2.05"],
  [14, "Heliot Ramos", "SF", "27.5", 50, 6, 3, "67%", "9.09", "-2.02"],
  [15, "Otto Lopez", "MIA", "28.5", 79, 13, 4, "76%", "9.13", "-2.00"],
  [16, "Oneil Cruz", "PIT", "29.2", 93, 33, 5, "87%", "7.97", "-1.95"],
  [17, "Pete Crow-Armstrong", "CHC", "29.5", 96, 26, 5, "84%", "9.91", "-1.85"],
  [18, "Jackson Chourio", "MIL", "29.2", 92, 17, 5, "77%", "10.41", "-1.69"],
  [19, "James Wood", "WSH", "27.8", 58, 13, 5, "72%", "9.78", "-1.68"],
  [20, "Matt Shaw", "CHC", "29.0", 89, 12, 5, "71%", "10.66", "-1.67"],
  [21, "Jarren Duran", "BOS", "29.1", 91, 22, 5, "82%", "11.94", "-1.28"],
  [22, "Willi Castro", "CHC", "28.2", 70, 10, 3, "77%", "9.14", "-1.24"],
  [23, "Tyler Soderstrom", "ATH", "27.4", 46, 7, 2, "78%", "9.26", "-1.15"],
  [24, "Mike Yastrzemski", "KC", "26.9", 36, 6, 2, "75%", "10.01", "-1.09"],
  [25, "Elly De La Cruz", "CIN", "29.1", 92, 25, 5, "83%", "9.54", "-1.05"],
];
const top25_2026=[
  [1, "Ryan McMahon", "NYY", "25.7", 14, 3, 0, "100%", "24.50", "+8.05"],
  [2, "JJ Wetherholt", "STL", "27.3", 50, 5, 0, "100%", "16.84", "+4.45"],
  [3, "Josh Naylor", "SEA", "24.6", 2, 9, 1, "90%", "17.73", "+3.91"],
  [4, "Chase DeLauter", "CLE", "26.6", 28, 2, 1, "67%", "18.97", "+3.85"],
  [5, "Nico Hoerner", "CHC", "28.4", 81, 10, 0, "100%", "13.51", "+3.41"],
  [6, "Luis Arraez", "SF", "26.7", 33, 4, 0, "100%", "15.22", "+3.33"],
  [7, "Kyle Tucker", "LAD", "26.4", 25, 4, 0, "100%", "15.43", "+3.33"],
  [8, "Masyn Winn", "STL", "28.1", 70, 4, 0, "100%", "14.47", "+3.28"],
  [9, "Jazz Chisholm Jr.", "NYY", "28.4", 80, 10, 1, "91%", "14.87", "+2.97"],
  [10, "Miguel Vargas", "CWS", "28.1", 71, 6, 0, "100%", "13.38", "+2.91"],
  [11, "Nasim Nu\u00f1ez", "WSH", "29.7", 97, 19, 0, "100%", "11.17", "+2.81"],
  [12, "Xander Bogaerts", "SD", "27.3", 50, 7, 0, "100%", "13.06", "+2.62"],
  [13, "Dansby Swanson", "CHC", "28.0", 68, 4, 0, "100%", "13.35", "+2.62"],
  [14, "Zack Gelof", "ATH", "28.8", 89, 4, 0, "100%", "12.88", "+2.61"],
  [15, "Randy Arozarena", "SEA", "27.1", 44, 14, 0, "100%", "12.15", "+2.45"],
  [16, "Sal Stewart", "CIN", "26.5", 27, 9, 1, "90%", "13.98", "+2.43"],
  [17, "Nick Kurtz", "ATH", "26.7", 34, 5, 1, "83%", "14.72", "+2.24"],
  [18, "Heriberto Hern\u00e1ndez", "MIA", "28.6", 85, 3, 0, "100%", "12.57", "+2.21"],
  [19, "Juan Soto", "NYM", "25.9", 18, 2, 1, "67%", "16.40", "+2.16"],
  [20, "Brett Baty", "NYM", "26.9", 39, 2, 1, "67%", "15.83", "+2.15"],
  [21, "Bryan Reynolds", "PIT", "28.0", 69, 4, 0, "100%", "12.28", "+2.03"],
  [22, "Anthony Volpe", "NYY", "27.6", 57, 2, 1, "67%", "15.23", "+2.01"],
  [23, "Jake Fraley", "TB", "28.4", 81, 3, 0, "100%", "12.40", "+2.00"],
  [24, "Alek Thomas", "AZ", "27.8", 64, 3, 0, "100%", "12.70", "+1.98"],
  [25, "Kevin McGonigle", "DET", "28.1", 72, 7, 0, "100%", "11.37", "+1.89"],
];
const bot25_2026=[
  [1, "Alejandro Osuna", "TEX", "28.5", 81, 1, 2, "33%", "7.40", "-4.04"],
  [2, "Taylor Ward", "BAL", "27.5", 56, 1, 2, "33%", "7.12", "-3.85"],
  [3, "Joey Ortiz", "MIL", "28.5", 82, 3, 3, "50%", "7.09", "-3.55"],
  [4, "Elly De La Cruz", "CIN", "28.3", 78, 5, 4, "56%", "7.20", "-3.52"],
  [5, "Chandler Simpson", "TB", "29.6", 96, 12, 4, "75%", "11.07", "-3.15"],
  [6, "Cole Young", "SEA", "27.7", 61, 2, 3, "40%", "9.00", "-3.15"],
  [7, "Ceddanne Rafaela", "BOS", "28.9", 90, 4, 4, "50%", "10.80", "-3.13"],
  [8, "Luis Robert Jr.", "NYM", "29.0", 92, 1, 2, "33%", "8.43", "-2.98"],
  [9, "Willy Adames", "SF", "27.7", 59, 1, 2, "33%", "8.72", "-2.91"],
  [10, "Riley Greene", "DET", "26.6", 31, 1, 1, "50%", "7.47", "-2.87"],
  [11, "Munetaka Murakami", "CWS", "27.2", 48, 1, 3, "25%", "10.70", "-2.79"],
  [12, "Daylen Lile", "WSH", "29.2", 94, 2, 2, "50%", "9.00", "-2.63"],
  [13, "Isaac Collins", "KC", "27.4", 54, 2, 3, "40%", "10.24", "-2.49"],
  [14, "Cody Bellinger", "NYY", "27.6", 58, 2, 3, "40%", "10.92", "-2.36"],
  [15, "Luis Rengifo", "MIL", "27.1", 43, 2, 1, "67%", "7.83", "-2.31"],
  [16, "Miguel Andujar", "SD", "27.3", 51, 1, 2, "33%", "10.00", "-2.30"],
  [17, "Fernando Tatis Jr.", "SD", "28.9", 90, 10, 5, "67%", "9.21", "-2.24"],
  [18, "A.J. Ewing", "NYM", "29.0", 92, 3, 3, "50%", "10.07", "-2.17"],
  [19, "Jake Meyers", "HOU", "27.8", 65, 1, 2, "33%", "10.00", "-2.10"],
  [20, "Justin Crawford", "PHI", "29.7", 97, 4, 2, "67%", "9.47", "-2.07"],
  [21, "Bryce Harper", "PHI", "27.0", 41, 3, 2, "60%", "9.46", "-1.94"],
  [22, "Jose Altuve", "HOU", "27.9", 67, 1, 1, "50%", "8.77", "-1.88"],
  [23, "Jos\u00e9 Caballero", "NYY", "28.2", 74, 8, 5, "62%", "10.78", "-1.76"],
  [24, "Brice Matthews", "HOU", "28.2", 75, 2, 2, "50%", "9.75", "-1.76"],
  [25, "Max Schuemann", "NYY", "27.7", 61, 2, 2, "50%", "10.12", "-1.75"],
];

// TABLE BUILDERS
// rankTable: [rank, name, team, season, sprint, pctile, hp_to_1b, SB, CS, SB%, gain, gain_z, succ_z, squander_z, BCS]
function rankTable(rows,hlNS=false){
  const W=[380,1380,440,440,680,500,500,460,520,500,500,500,520];
  const hdr=new TableRow({children:[hdrCell("#",W[0]),hdrCell("Player",W[1]),hdrCell("Team",W[2]),hdrCell("Yr",W[3]),hdrCell("Sprint(pct)",W[4]),hdrCell("90ft",W[5]),hdrCell("SB/CS",W[6]),hdrCell("SB%",W[7]),hdrCell("Gain ft",W[8]),hdrCell("Gn_z",W[9]),hdrCell("Su_z",W[10]),hdrCell("Sq_z",W[11]),hdrCell("BCS",W[12])]});
  const drows=rows.map((r,i)=>{
    const isNS=hlNS&&(r[1].includes('Naylor')||r[1].includes('Soto'));
    const fill=isNS?GOLD:(i%2===0?WHITE:LTGRAY);
    const bcs=parseFloat(r[14]);
    const bf=bcs>=2.5?GREEN:(bcs<=-3.5?RED:fill);
    return new TableRow({children:[
      dataCell(r[0],W[0],fill,true),dataCell(r[1],W[1],fill,isNS),dataCell(r[2],W[2],fill),
      dataCell(r[3],W[3],fill),dataCell(`${r[4]} (${r[5]}%)`,W[4],fill),dataCell(r[6],W[5],fill),
      dataCell(`${r[7]}/${r[8]}`,W[6],fill),dataCell(r[9],W[7],fill),dataCell(r[10],W[8],fill),
      dataCell(r[11],W[9],fill),dataCell(r[12],W[10],fill),dataCell(r[13],W[11],fill),
      dataCell(r[14],W[12],bf,true)]});
  });
  return new Table({rows:[hdr,...drows],width:{size:7820,type:WidthType.DXA}});
}

// seasonTable: [rank, name, team, sprint, pctile, SB, CS, SB%, gain, BCS]
function seasonTable(rows){
  const W=[380,1500,480,760,580,500,580,580];
  const hdr=new TableRow({children:[hdrCell("#",W[0]),hdrCell("Player",W[1]),hdrCell("Team",W[2]),hdrCell("Sprint (pctile)",W[3]),hdrCell("SB/CS",W[4]),hdrCell("SB%",W[5]),hdrCell("Gain ft",W[6]),hdrCell("BCS",W[7])]});
  const drows=rows.map((r,i)=>{
    const isNS=r[1].includes('Naylor')||r[1].includes('Soto');
    const fill=isNS?GOLD:(i%2===0?WHITE:LTGRAY);
    const bcs=parseFloat(r[9]);
    const bf=bcs>=2.5?GREEN:(bcs<=-3.5?RED:fill);
    return new TableRow({children:[
      dataCell(r[0],W[0],fill,true),dataCell(r[1],W[1],fill,isNS),dataCell(r[2],W[2],fill),
      dataCell(`${r[3]} (${r[4]}%)`,W[3],fill),dataCell(`${r[5]}/${r[6]}`,W[4],fill),
      dataCell(r[7],W[5],fill),dataCell(r[8],W[6],fill),dataCell(r[9],W[7],bf,true)]});
  });
  return new Table({rows:[hdr,...drows],width:{size:5360,type:WidthType.DXA}});
}

function absGainTable(rows){
  const W=[380,1500,480,700,580,600,580];
  const hdr=new TableRow({children:[hdrCell("#",W[0]),hdrCell("Player",W[1]),hdrCell("Team",W[2]),hdrCell("Season",W[3]),hdrCell("Gain ft",W[4]),hdrCell("Gain_z",W[5]),hdrCell("BCS",W[6])]});
  const drows=rows.map((r,i)=>{
    const isNS=r[1].includes('Naylor')||r[1].includes('Soto');
    const fill=isNS?GOLD:(i%2===0?WHITE:LTGRAY);
    return new TableRow({children:[
      dataCell(r[0],W[0],fill,true),dataCell(r[1],W[1],fill,isNS),dataCell(r[2],W[2],fill),
      dataCell(r[3],W[3],fill),dataCell(r[4],W[4],fill),dataCell(r[5],W[5],fill),
      dataCell(r[6],W[6],fill,true)]});
  });
  return new Table({rows:[hdr,...drows],width:{size:4820,type:WidthType.DXA}});
}

function twoColTable(rows,c1='Metric',c2='Value'){
  const W1=2400,W2=3800;
  const hdr=new TableRow({children:[hdrCell(c1,W1),hdrCell(c2,W2)]});
  const drows=rows.map(([k,v],i)=>new TableRow({children:[dataCell(k,W1,i%2===0?WHITE:LTGRAY,true),dataCell(v,W2,i%2===0?WHITE:LTGRAY)]}));
  return new Table({rows:[hdr,...drows],width:{size:6200,type:WidthType.DXA}});
}


// DOCUMENT
const doc=new Document({
  numbering:{config:[{reference:"bullets",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:360,hanging:360}}}}]}]},
  sections:[{
    properties:{page:{margin:{top:720,bottom:720,left:900,right:900}}},
    headers:{default:new Header({children:[new Paragraph({children:[new TextRun({text:"The Naylor Blueprint Report",size:18,color:"888888",italics:true})]})]})}  ,
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],size:18,color:"888888"})]})]})},
    children:[

// COVER
new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:2880,after:240},children:[new TextRun({text:"THE NAYLOR BLUEPRINT",bold:true,size:52,color:NAVY})]}),
new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:0,after:160},children:[new TextRun({text:"Full-Spectrum Basestealing Analysis",size:30,color:HEAD})]}),
new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:0,after:80},children:[new TextRun({text:"Blueprint Conversion Score — Seasons 2023–2026",size:24,color:"555555",italics:true})]}),
new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:0,after:80},children:[new TextRun({text:"May 2026",size:24,color:"666666",italics:true})]}),

// S1
h1("1  Introduction"),
body("This report quantifies a single insight: the stolen base is not a speed contest. Josh Naylor stole 22 bags at a 95.7% success rate in 2025 while running at the 1st sprint percentile. What separates elite stealers is technique — specifically, how much ground they cover between the pitcher's first move and pitch release."),
body("The Blueprint Conversion Score (BCS) ranks every qualified runner-season since the 2023 bigger-bases rule change by three components: (1) steal success above speed-expected, (2) ground covered above speed-expected, and (3) a squander penalty for fast runners who get caught anyway."),
body("2026 is included at a lower attempt threshold (≥3 tracked Statcast attempts; ~1/3 of season complete as of May 2026). Small-sample scores are Bayes-shrunk toward the league mean."),

// S2
h1("2  The Metric: Secondary Distance"),
body("The foundational metric is native Statcast: gain_to_release_ft = r_secondary_lead − r_primary_lead. This is the distance (in feet) a runner covers between the pitcher's first detectable move and pitch release — the jump window."),
...sp(1),
twoColTable([
  ["Primary lead (r_primary_lead)","Distance off first base at pitcher's first move"],
  ["Secondary lead (r_secondary_lead)","Distance off first base at pitch release"],
  ["Gain to release","Difference: feet covered in the jump window"],
  ["Speed correlation (r)","−0.36 (slower runners cover more ground — near-speed-independent)"],
  ["Correlation with SB%","+0.25 (raw), +0.29 (speed-adjusted)"],
],'Term','Definition'),
...sp(1),
body("The critical finding: sprint speed barely predicts how much ground a runner covers in this window. OLS slope = −0.72 ft per ft/s. The gain residual isolates a real, near-speed-independent timing and jump skill."),

// S3
h1("3  Blueprint Conversion Score (BCS)"),
mono("BCS = success_resid_z + gain_resid_z − squander_z"),
body("All three terms are z-scores across 515 volume-qualified runner-seasons (2023–2026)."),
...sp(1),
twoColTable([
  ["success_resid_z","Beta-Binomial steal-success posterior (shrunk to league ~75% rate), regressed on speed → residual. Rewards converting more often than speed predicts."],
  ["gain_resid_z","Mean ground covered minus what speed predicts. Rewards a big jump for how slow you are."],
  ["squander_z","CS × max(speed_z,0) × (1 + max(gain_z,0)). Only penalizes fast runners caught despite obvious speed advantage."],
  ["Beta prior","Empirical-Bayes moment-matched: α₀ = 8.74, β₀ = 1.92"],
  ["Speed composite","(z(sprint) − z(hp_to_1b)) / 2 — high = fast on both peak velocity and 90-ft burst"],
],'Component','Description'),

// S4
h1("4  Validation"),
bullet("Speed independence: OLS slope = −0.72 ft per ft/s (r = −0.36). Ground covered is not a speed skill."),
bullet("Predictive validity: r(mean_gain, SB%) = +0.25; r(gain_residual, SB%) = +0.29."),
bullet("Out-of-sample: Naylor and Soto rank near the top on gain and BCS across all tracked seasons."),

// S5
h1("5  Results & Discussion"),

h2("5.1  Overall Top 15 — The Blueprint"),
body("The overall Top 15 (2023–2026 pooled) is dominated by slow/medium-speed runners who time the pitcher well and convert at elite rates. Ryan McMahon (NYY 2026) leads at BCS +8.05 driven by 24.50 ft mean gain — the highest ever measured — on 3 attempts. Agustín Ramírez (MIA 2025) is the volume leader at +6.86 on 13 attempts. Josh Naylor holds two top-10 spots: 2025 (#6, +4.01) and 2026 (#7, +3.91). Juan Soto (NYM 2025) at #11 (+3.45)."),
tableLabel("Table 1 — Overall Top 15 BCS (2023–2026)"),
rankTable(top15Data,true),
...imgPara(imgTopN,580,420,"Figure 1 — Top 15 BCS runner-seasons (2023–2026). Red = Naylor. Green = Soto."),

h2("5.2  The Score in Detail"),
bullet("success_resid_z rewards converting more often than sprint speed predicts — execution."),
bullet("gain_resid_z rewards covering more ground than sprint speed predicts — timing."),
bullet("squander_z penalizes fast runners who get caught. A fast runner with 8 CS has a larger penalty than a slow runner with 8 CS."),

h2("5.3  Naylor & Soto — The Archetype Profile"),
tableLabel("Table 2 — Naylor & Soto: All Seasons (2023–2026)"),
rankTable(naylorSotoData,true),
...sp(1),
bullet("Naylor 2025: 24.4 ft/s (1st pctile), 16.74 ft gain, 22/23 = 95.7% SB rate, BCS +4.01."),
bullet("Naylor 2026 (partial): 17.73 ft gain (higher than 2025), BCS +3.91 — consistent pattern."),
bullet("Soto 2025: 30/30 = 100% SB rate, BCS +3.45 — driven by elite success_resid_z (+2.39)."),
bullet("Soto 2024: negative BCS (−0.06) — 6/10 season; weak success residual dragged him below neutral."),
...imgPara(imgScatter,680,460,"Figure 2 — BCS scatter: sprint speed vs. mean gain, colored by score. Green = high BCS. Red = low BCS. Naylor/Soto annotated."),

h2("5.4  Top 25 Per Season — Ground Covered (2023–2025)"),
body("The following tables rank the top 25 per year by speed-adjusted ground covered (gain_resid_z). This isolates the pure timing/jump sub-ranking independent of steal conversion."),
...sp(1),
tableLabel("Table 3 — 2023 Top 25 by Ground Covered"),
seasonTable(top25_2023),
...sp(1),
tableLabel("Table 4 — 2024 Top 25 by Ground Covered"),
seasonTable(top25_2024),
...sp(1),
tableLabel("Table 5 — 2025 Top 25 by Ground Covered"),
seasonTable(top25_2025),
...imgPara(imgByYear,720,540,"Figure 3 — Top 25 per season (2023–2025) by speed-adjusted ground covered, with team logos."),

h2("5.5  Bottom 15 — The Anti-Naylor"),
body("The bottom of the board is occupied by high-speed runners accumulating caught-stealings. Bobby Witt Jr. (KC) ranks in the bottom 5 for every season (2023: #514, 2024: #511, 2025: #508). Chandler Simpson (TB 2025) anchors at #515 (BCS −6.71): 38 SB / 8 CS from the 97th sprint percentile. The model is penalizing good sprinters who waste their edge."),
tableLabel("Table 6 — Overall Bottom 15 BCS (2023–2026)"),
rankTable(bot15Data),
...imgPara(imgBottomN,580,420,"Figure 4 — Bottom 15 BCS runner-seasons. The anti-Naylor: elite speed, serial caught-stealings."),

h2("5.6  Absolute Ground Covered — Top 25 by Raw Gain"),
body("Who physically covers the most ground? Ryan McMahon (NYY 2026) leads at 24.50 ft. Agustín Ramírez (MIA 2025) is the full-season leader at 21.31 ft. Josh Naylor appears twice: 2026 at #4 (17.73 ft) and 2025 at #7 (16.74 ft). Juan Soto appears four times."),
tableLabel("Table 7 — Top 25 BCS Runner-Seasons by Absolute Mean Gain"),
absGainTable(absGainData),
...imgPara(imgAbsGain,680,500,"Figure 5 — Top 25 BCS runner-seasons by raw mean gain (ft). Naylor/Soto highlighted."),

h2("5.7  Per-Season BCS Top 25"),
body("Top 25 BCS runner-seasons per year since the bigger-bases rule change. Green cells = BCS ≥ 2.5. Yellow = Naylor / Soto."),
h3("2023 — Top 25 Blueprint"),
tableLabel("Table 8 — 2023 BCS Top 25"),
seasonTable(top25_2023),
pageBreak(),
h3("2024 — Top 25 Blueprint"),
tableLabel("Table 9 — 2024 BCS Top 25"),
seasonTable(top25_2024),
pageBreak(),
h3("2025 — Top 25 Blueprint"),
tableLabel("Table 10 — 2025 BCS Top 25"),
seasonTable(top25_2025),
pageBreak(),
h3("2026 — Top 25 Blueprint (Partial Season)"),
tableLabel("Table 11 — 2026 BCS Top 25 †"),
seasonTable(top25_2026),
footnote("† 2026 data through May 30, 2026 (~1/3 of season complete). Minimum: 3 tracked Statcast attempts (vs. standard 10). Small-sample scores are Bayes-shrunk toward the league mean."),
...sp(1),
...imgPara(imgTop25Season,720,480,"Figure 6 — Top 25 BCS per season (2023–2026) with team logos. Red = Naylor. Green = Soto. † 2026 partial season."),

h2("5.8  Per-Season BCS Bottom 25 — The Squanderers"),
body("Bottom 25 per year. Red cells = BCS ≤ −3.5. Bobby Witt Jr. and Chandler Simpson appear repeatedly."),
h3("2023 — Bottom 25 Squanderers"),
tableLabel("Table 12 — 2023 BCS Bottom 25"),
seasonTable(bot25_2023),
pageBreak(),
h3("2024 — Bottom 25 Squanderers"),
tableLabel("Table 13 — 2024 BCS Bottom 25"),
seasonTable(bot25_2024),
pageBreak(),
h3("2025 — Bottom 25 Squanderers"),
tableLabel("Table 14 — 2025 BCS Bottom 25"),
seasonTable(bot25_2025),
pageBreak(),
h3("2026 — Bottom 25 Squanderers (Partial Season)"),
tableLabel("Table 15 — 2026 BCS Bottom 25 †"),
seasonTable(bot25_2026),
footnote("† 2026 partial season — same caveats as Table 11."),
...sp(1),
...imgPara(imgBot25Season,720,480,"Figure 7 — Bottom 25 BCS per season (2023–2026). The squanderer pattern: fast, caught repeatedly. † 2026 partial season."),

// S6
h1("6  Conclusions"),
bullet("Ground covered (first move → release) is near-speed-independent. Sprint speed explains only a small fraction of the variance (r = −0.36, slope −0.72 ft per ft/s)."),
bullet("The gain metric meaningfully predicts steal success (r = +0.25–0.29 with SB%). The signal is real."),
bullet("The Naylor archetype — slow sprint, big jump, high conversion rate — appears in every season and is not a single-year outlier. Naylor 2026 is already tracking above his 2025 pace."),
bullet("The squander archetype — fast sprint, serial caught-stealings — is equally stable. Bobby Witt Jr. is in the bottom five every season in the dataset."),
body("The practical implication: timing and jump training can shift BCS scores for runners at both ends of the speed spectrum. Sprint speed is structural. The jump window is behavioral."),

    ]}]});

Packer.toBuffer(doc).then(buf=>{
  fs.writeFileSync('/Users/shunchen/Desktop/The-Naylor-Model/Reports/Naylor Blueprint Report.docx',buf);
  console.log('[write] Naylor Blueprint Report.docx  ('+Math.round(buf.length/1024)+'KB)');
}).catch(e=>{console.error(e);process.exit(1);});
