// Build the K-Health preliminary-round proposal as a .docx.
//
//   npm install docx && node scripts/build_proposal.js
//
// The prose lives in docs/proposal-draft.md — edit there first, then mirror the
// change here; this file is the typeset copy, not a second source of truth. The
// figures come from `python src/vitals_report.py` and `src/vitals_phenotype.py`,
// which write into models/ and run on the built-in synthetic cohort, so a fresh
// clone can rebuild the document without any data.
//
// PAGES below holds the table-of-contents page numbers. They are read off the
// rendered PDF rather than computed, so re-check them after editing content:
//
//   soffice --headless --convert-to pdf models/PRODROME_제안서.docx
//   for p in $(seq 1 24); do echo "p$p: $(pdftotext -f $p -l $p ... - | head -2)"; done
//
// Output goes to models/ (gitignored) unless PROPOSAL_OUT says otherwise.
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, ImageRun,
  Footer, PageNumber, convertInchesToTwip, Tab, TabStopType, LeaderType,
} = require("docx");
const fs = require("fs");

const REPO = require("path").resolve(__dirname, "..");
const OUT = process.env.PROPOSAL_OUT ?? `${REPO}/models/PRODROME_제안서.docx`;

const FONT = "맑은 고딕";
const INK = "1A1A1A", MUTED = "555555", ACCENT = "1F4E79", WARN = "9C2B2B";
const CONTENT_W = 9026;               // A4 width 11906 - 2*1440 margins

// ─────────────────────────────────────────── helpers
const P = (text, o = {}) => new Paragraph({
  alignment: o.align,
  spacing: { before: o.before ?? 0, after: o.after ?? 140, line: o.line ?? 300 },
  indent: o.indent,
  border: o.border,
  shading: o.shading,
  children: (Array.isArray(text) ? text : [{ t: text }]).map((r) => new TextRun({
    text: r.t,
    bold: r.b ?? o.bold,
    italics: r.i ?? o.italics,
    color: r.c ?? o.color ?? INK,
    size: r.s ?? o.size ?? 20,          // half-points → 10pt
    font: FONT,
  })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 200 },
  children: [new TextRun({ text, bold: true, size: 30, color: ACCENT, font: FONT })],
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 260, after: 140 },
  children: [new TextRun({ text, bold: true, size: 24, color: INK, font: FONT })],
});

const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 200, after: 110 },
  children: [new TextRun({ text, bold: true, size: 21, color: ACCENT, font: FONT })],
});

// bullet without a numbering config — a real glyph run is fragile, so use indent + en-dash
const BULLET = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 90, line: 300 },
  indent: { left: 360, hanging: 200 },
  children: [
    new TextRun({ text: "· ", bold: true, size: 20, color: ACCENT, font: FONT }),
    ...(Array.isArray(text) ? text : [{ t: text }]).map((r) => new TextRun({
      text: r.t, bold: r.b, italics: r.i, color: r.c ?? INK, size: 20, font: FONT,
    })),
  ],
});

const NUM = (n, text) => new Paragraph({
  spacing: { after: 110, line: 300 },
  indent: { left: 420, hanging: 260 },
  children: [
    new TextRun({ text: `${n}. `, bold: true, size: 20, color: ACCENT, font: FONT }),
    ...(Array.isArray(text) ? text : [{ t: text }]).map((r) => new TextRun({
      text: r.t, bold: r.b, italics: r.i, color: r.c ?? INK, size: 20, font: FONT,
    })),
  ],
});

/** Callout box — left rule + tint, for caveats and definitions. */
const CALLOUT = (runs, tint = "F2F6FA", rule = ACCENT) => new Paragraph({
  spacing: { before: 160, after: 180, line: 300 },
  indent: { left: 200, right: 200 },
  shading: { type: ShadingType.CLEAR, fill: tint },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: rule, space: 10 } },
  children: (Array.isArray(runs) ? runs : [{ t: runs }]).map((r) => new TextRun({
    text: r.t, bold: r.b, italics: r.i, color: r.c ?? INK, size: 19, font: FONT,
  })),
});

const cell = (runs, o = {}) => new TableCell({
  width: { size: o.w, type: WidthType.DXA },
  shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
  margins: { top: 90, bottom: 90, left: 130, right: 130 },
  verticalAlign: "center",
  children: [new Paragraph({
    alignment: o.align,
    spacing: { after: 0, line: 260 },
    children: (Array.isArray(runs) ? runs : [{ t: String(runs) }]).map((r) => new TextRun({
      text: r.t, bold: r.b ?? o.bold, italics: r.i,
      color: r.c ?? o.color ?? INK, size: r.s ?? 18, font: FONT,
    })),
  })],
});

/** rows: array of arrays; widths must sum to CONTENT_W */
const TABLE = (widths, header, rows, opts = {}) => new Table({
  columnWidths: widths,
  width: { size: CONTENT_W, type: WidthType.DXA },
  borders: {
    top:    { style: BorderStyle.SINGLE, size: 4, color: "AAB4BE" },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: "AAB4BE" },
    left:   { style: BorderStyle.SINGLE, size: 4, color: "AAB4BE" },
    right:  { style: BorderStyle.SINGLE, size: 4, color: "AAB4BE" },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "CCD4DC" },
    insideVertical:   { style: BorderStyle.SINGLE, size: 2, color: "CCD4DC" },
  },
  rows: [
    new TableRow({
      tableHeader: true,
      children: header.map((h, i) => cell(h, {
        w: widths[i], fill: "E8EEF5", bold: true, color: ACCENT,
        align: opts.headerAlign?.[i],
      })),
    }),
    ...rows.map((r) => new TableRow({
      cantSplit: true,
      children: r.map((c, i) => cell(c, { w: widths[i], align: opts.align?.[i] })),
    })),
  ],
});

const FIG = (file, ratio, caption, widthPx = 560) => !fs.existsSync(`${REPO}/models/${file}`) ? [
  // Figures live in the gitignored models/; a fresh clone regenerates them with
  // vitals_report.py / vitals_phenotype.py. Leave a marker rather than crash.
  P(`〔그림 누락: models/${file} — vitals_report.py 실행 필요〕`, {
    align: AlignmentType.CENTER, color: WARN, italics: true, size: 18,
  }),
] : [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 80 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(`${REPO}/models/${file}`),
      transformation: { width: widthPx, height: Math.round(widthPx * ratio) },
    })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 220 },
    children: [new TextRun({ text: caption, size: 17, color: MUTED, italics: true, font: FONT })],
  }),
];

const BREAK = () => new Paragraph({ children: [new PageBreak()] });
const GAP = (n = 200) => new Paragraph({ spacing: { after: n }, children: [] });

/** Static TOC line: label, dot leader to a right tab stop, then the page number. */
const TOC = (label, page, o = {}) => new Paragraph({
  spacing: { after: o.sub ? 70 : 110, line: 280 },
  indent: { left: o.sub ? 340 : 0 },
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: LeaderType.DOT }],
  children: [
    new TextRun({
      text: label,
      bold: !o.sub,
      size: o.sub ? 19 : 21,
      color: o.sub ? MUTED : INK,
      font: FONT,
    }),
    new TextRun({
      font: FONT, size: o.sub ? 19 : 21, color: o.sub ? MUTED : INK,
      children: [new Tab(), String(page)],
    }),
  ],
});

// ─────────────────────────────────────────── content
const children = [];

// ── 표지
children.push(
  GAP(1600),
  P("2026 K-Health 미개방 의료데이터 활용 경진대회", {
    align: AlignmentType.CENTER, color: MUTED, size: 22, after: 100,
  }),
  P("예선 아이디어 제안서", {
    align: AlignmentType.CENTER, color: MUTED, size: 20, after: 700,
  }),
  P("개인화 기저선 이탈 기반", {
    align: AlignmentType.CENTER, bold: true, size: 40, color: ACCENT, after: 90,
  }),
  P("설명가능 원내 심정지 조기경보", {
    align: AlignmentType.CENTER, bold: true, size: 40, color: ACCENT, after: 560,
  }),
  P([{ t: "심정지가 남기는 " }, { t: "전조", b: true },
     { t: "를 환자 자신의 기저선 기준으로 포착해," }], {
    align: AlignmentType.CENTER, size: 22, after: 60,
  }),
  P("왜 위험한지와 함께 알리는 조기경보", {
    align: AlignmentType.CENTER, size: 22, after: 900,
  }),
);

children.push(TABLE(
  [2400, 6626],
  ["항목", "내용"],
  [
    ["팀명", [{ t: "PRODROME (전조)", b: true }]],
    ["팀 구성", "1인 팀"],
    ["선택 데이터셋", [{ t: "[대구] 경북대학교병원 입원환자 활력징후(Vital Sign) 데이터", b: true }]],
    ["활용 테이블", "KHTH_PINFO · KHTH_VITAL"],
    ["제출일", "2026년 8월 13일"],
  ],
));

children.push(
  GAP(400),
  P("본 제안서에 수록된 예비 수치의 출처(공개 실데이터 / 합성 데이터)는 제7장에 구분하여 명시하였습니다.", {
    align: AlignmentType.CENTER, color: MUTED, size: 17,
  }),
  BREAK(),
);

// ── 목차   (page numbers filled from the rendered PDF — see PAGES below)
const PAGES = {
  summary: 3, s1: 4, s13: 4, s2: 5, s3: 7, s4: 8, s5: 9, s52: 9, s53: 10, s6: 11,
  s7: 13, s71: 13, s72: 14, s73: 17, s74: 19,
  s8: 20, s9: 21, s96: 22, s10: 23, s11: 24, ref: 25,
};
children.push(
  H1("목차"),
  GAP(160),
  TOC("요약", PAGES.summary),
  TOC("1. 배경 및 필요성", PAGES.s1),
  TOC("1.3 지금이 이 문제를 다룰 시점인 이유", PAGES.s13, { sub: true }),
  TOC("2. 문제 정의 및 목표", PAGES.s2),
  TOC("3. 기존 접근과 한계", PAGES.s3),
  TOC("4. 제안 아이디어", PAGES.s4),
  TOC("5. 데이터 활용 방안", PAGES.s5),
  TOC("5.2 case-only 구조에서 오경보를 어떻게 측정할 것인가", PAGES.s52, { sub: true }),
  TOC("5.3 데이터 품질 점검 및 정제 규칙", PAGES.s53, { sub: true }),
  TOC("6. 분석 방법론", PAGES.s6),
  TOC("7. 검증 결과", PAGES.s7),
  TOC("7.1 공개 실데이터 검증 — Challenge 2019 〔실측〕", PAGES.s71, { sub: true }),
  TOC("7.2 심정지 파이프라인 작동 시연 〔합성 데이터〕", PAGES.s72, { sub: true }),
  TOC("7.3 인용 시 반드시 함께 밝히는 조건", PAGES.s73, { sub: true }),
  TOC("7.4 설명가능성 실증 — 근거가 화면에 도달하는가", PAGES.s74, { sub: true }),
  TOC("8. 기대효과 및 파급성", PAGES.s8),
  TOC("9. 추진 계획 및 안심존 활용", PAGES.s9),
  TOC("9.6 위험 요인 및 대응 계획", PAGES.s96, { sub: true }),
  TOC("10. 윤리 · 개인정보 · 데이터 보안", PAGES.s10),
  TOC("11. 결론", PAGES.s11),
  TOC("참고문헌", PAGES.ref),
  BREAK(),
);

// ── 요약
children.push(
  H1("요약"),
  P([
    { t: "무엇을 하는가. " , b: true },
    { t: "입원 환자의 활력징후 시계열을 입력으로, 향후 단기간 내 원내 심정지 발생 위험을 시점마다 산출하고, 임계값을 넘으면 " },
    { t: "그 판단의 근거와 함께", b: true },
    { t: " 경보한다. 의료진이 신속대응팀 호출·중환자실 이송 등 선제 조치를 취할 수 있도록 돕는 것이 목적이다." },
  ]),
  P([
    { t: "왜 정확도가 아니라 오경보인가. ", b: true },
    { t: "활력징후로 심정지를 예측하는 것은 이미 상용 AI 의료기기로 구현되어 임상적·규제적으로 입증된 영역이다. 그럼에도 병동의 조기경보가 신뢰받지 못하는 이유는 정확도가 아니라 " },
    { t: "오경보와 근거 부재", b: true },
    { t: "다. 잦은 거짓 알람은 alarm fatigue를 유발해 필요한 경보까지 무시되게 만들고, 점수만 제시하는 경보는 임상 판단을 지원하지 못한다. 본 제안은 “더 잘 맞히는가”가 아니라 " },
    { t: "“같은 검출률에서 얼마나 적게, 얼마나 일찍, 왜 알리는가”", b: true },
    { t: "를 성공 기준으로 삼는다." },
  ]),
  H3("세 가지 차별점"),
  NUM(1, [
    { t: "개인화 기저선 이탈. ", b: true },
    { t: "병동 공통 임계값이 아니라 각 환자의 초기 안정기를 개인 기저선으로 삼아 그로부터의 이탈을 위험신호로 사용한다. 환자가 곧 자기 자신의 대조군이 된다." },
  ]),
  NUM(2, [
    { t: "설명가능성 내장. ", b: true },
    { t: "모든 경보에 대해 어떤 활력징후가 어느 방향으로 기여했는지를 SHAP 기반으로 함께 제시한다." },
  ]),
  NUM(3, [
    { t: "case-only 구조의 방법론적 전환. ", b: true },
    { t: "대조군이 없는 데이터 구조를 한계로 두지 않고 within-patient 설계의 근거로 활용하며, 그로 인한 측정의 한계까지 명시하고 보정 경로를 설계에 포함한다." },
  ]),
  H3("실현성 근거"),
  P("동일 파이프라인을 공개 실데이터(PhysioNet/CinC Challenge 2019, 20,336명)에 적용해 적재·정제·윈도우 생성·학습·평가·설명까지 end-to-end 동작을 확인하였다. 검출률을 고정한 비교에서 임상 표준 NEWS 대비 알람 부담이 낮아지는 것을 실측하였으며(제7장), 근거 제시 화면까지 구현·배포하였다."),
  CALLOUT([
    { t: "본 제안서의 수치 표기 원칙. ", b: true },
    { t: "제7-1장은 공개 실데이터에서 얻은 실측치이며, 대상 사건은 패혈증 발병이다. 제7-2장은 심정지 라벨 체계에서의 파이프라인 작동 시연으로 합성·개발 데이터 기반이며 실제 환자 성능이 아니다. 두 수치를 혼용하여 인용하지 않는다." },
  ]),
  BREAK(),
);

// ── 1. 배경
children.push(
  H1("1. 배경 및 필요성"),
  P([
    { t: "원내 심정지(In-Hospital Cardiac Arrest, IHCA)는 입원 환자에게 발생하는 가장 치명적인 급성 사건 중 하나다. 생존·퇴원율이 낮고, 생존하더라도 신경학적 후유장애가 크다. 그런데 다수의 심정지는 " },
    { t: "갑작스러운 사건이 아니다", b: true },
    { t: ". 수 시간 전부터 맥박·호흡수의 상승, 혈압·SpO₂의 저하와 같은 활력징후 악화 " },
    { t: "전조(prodrome)", b: true },
    { t: "를 동반한다. 이 전조를 제때 포착하면 신속대응팀 호출, 중환자실 조기 이송과 같은 선제 개입으로 결과를 바꿀 임상적 여지가 있다. 본 팀의 이름을 PRODROME으로 정한 것도 이 전제를 그대로 가리키기 위함이다." },
  ]),
  H2("1.1 병동은 이미 조기경보를 운영하고 있다 — 그러나 신뢰받지 못한다"),
  P("이러한 배경에서 병동은 조기경보점수(NEWS/MEWS)를 운영한다. 그러나 규칙 기반 점수는 모든 환자에게 동일한 고정 임계값을 적용하므로 개인차를 반영하지 못한다. 평소 호흡수가 22회인 환자와 14회인 환자에게 같은 기준을 적용하면, 전자에게는 상시 경보가 울리고 후자에게는 실제 악화가 늦게 포착된다."),
  P([
    { t: "더 근본적인 문제는 사건의 희귀성에 있다. 심정지는 극단적으로 드문 사건이므로, ROC-AUC가 높게 나오더라도 AUPRC는 낮게 유지된다. 즉 " },
    { t: "지표상으로는 우수해 보이는 모델이 실제 운영에서는 오경보를 대량으로 발생시킬 수 있다", b: true },
    { t: ". 그리고 잦은 거짓 알람은 alarm fatigue를 유발해, 정작 필요한 경보까지 무시되게 만든다. 성능이 우수한 모델조차 현장에 정착하지 못하는 핵심 원인이 여기에 있다." },
  ]),
  CALLOUT([
    { t: "alarm fatigue. ", b: true },
    { t: "경보가 지나치게 자주 울리면 의료진은 경보에 대한 반응을 점진적으로 중단한다. 이는 개인의 태만이 아니라 과다 자극에 대한 예측 가능한 인지 반응이며, 따라서 해결책도 교육이 아니라 " },
    { t: "경보 설계", b: true },
    { t: "의 영역에 있다." },
  ]),
  P([
    { t: "이는 추상적 우려가 아니라 정량적으로 보고된 문제다. 미국 보건의료연구품질청(AHRQ)의 환자안전 실무 분석에 따르면 중환자실 환자 1명은 간호사 한 교대 동안 평균 " },
    { t: "150~400건의 알람", b: true },
    { t: "을 발생시키며, 심전도 모니터 알람의 " },
    { t: "80~99%가 거짓이거나 임상적으로 무의미", b: true },
    { t: "한 것으로 보고된다[4]. 미국 병원인증기구(The Joint Commission)는 임상 알람 안전을 국가환자안전목표로 지정하고 있다. 즉 경보의 수를 줄이는 것은 편의의 문제가 아니라 환자안전의 문제다." },
  ]),
  H2("1.2 지금 필요한 것은 더 높은 정확도가 아니다"),
  P("고령화와 중증환자 증가, 의료 AI의 실사용 요구가 동시에 맞물리는 현 시점에서 필요한 것은 “더 높은 정확도”가 아니라 “임상이 신뢰하고 실제로 사용하는 경보”다. 이는 구체적으로 세 가지 조건으로 분해된다."),
  BULLET([{ t: "적은 오경보 ", b: true }, { t: "— 같은 검출 수준을 유지하면서 알람 횟수를 줄일 것" }]),
  BULLET([{ t: "근거 제시 ", b: true }, { t: "— 왜 이 경보가 울렸는지를 임상이 검증할 수 있을 것" }]),
  BULLET([{ t: "개인화 ", b: true }, { t: "— 환자별 기저 상태 차이를 반영할 것" }]),
  P("이 세 조건은 환자안전이라는 국민 건강 가치에 직접 기여하며, 본 대회가 지향하는 미개방 의료데이터의 실질적 활용 가치와도 부합한다."),
  H2("1.3 지금이 이 문제를 다룰 시점인 이유"),
  P("국내 상용 솔루션이 제도권에 진입하는 과정 자체가 본 제안의 문제의식을 그대로 보여준다."),
  P([
    { t: "VUNO Med-DeepCARS는 2021년 8월 식품의약품안전처 허가를 획득해 생체신호 기반 AI 의료기기로는 국내 최초로 상용화 절차에 진입하였고, 이후 " },
    { t: "신의료기술평가 유예(선진입 의료기술) 제도", b: true },
    { t: "를 통해 비급여로 사용되어 왔다. 그 유예 기간이 2026년 종료되면서, 현재 정식 신의료기술평가 절차가 진행 중이다." },
  ]),
  CALLOUT([
    { t: "이 관문에서 요구되는 것은 예측 정확도가 아니다. ", b: true },
    { t: "정식 평가와 급여 진입은 “실제 임상 현장에서 도입했을 때 환자 결과가 개선되는가”라는 유용성 근거로 판단된다. 기술적 가능성은 이미 논문으로 입증되어 있고[2][3], 남아 있는 질문은 " },
    { t: "그 기술이 병동에서 실제로 작동하는가", b: true },
    { t: "이다." },
  ]),
  P([
    { t: "본 제안이 정확도 경쟁 대신 오경보 저감과 근거 제시를 겨냥하는 이유가 여기에 있다. " },
    { t: "현장에서 쓰이지 않는 경보는 성능과 무관하게 환자 결과를 바꾸지 못한다.", b: true },
    { t: " 지금은 이 분야가 “예측할 수 있는가”에서 “실제로 쓰이게 만들 수 있는가”로 이동하는 국면이며, 본 과제는 후자를 직접 겨냥한다." },
  ]),
  P("〔인허가·제도 현황은 2026년 8월 기준 공개 자료에 근거하며, 제출 시점에 재확인이 필요하다.〕", { size: 18, color: MUTED }),
);

// ── 2. 문제 정의
children.push(
  H1("2. 문제 정의 및 목표"),
  H2("2.1 문제의 정식화"),
  P("본 과제는 입원 환자의 시점별 활력징후 6종(맥박, 수축기혈압, 이완기혈압, 체온, SpO₂, 호흡수)을 입력으로, 향후 단기간(예: 1시간) 내 심정지 발생 여부를 조기에, 적은 오경보로, 근거와 함께 경보하는 이진 조기경보 문제로 정식화한다."),
  P([
    { t: "학습 단위는 각 환자의 시간축을 따라 슬라이딩하는 관찰 윈도우다. 심정지 시각(CARDT)을 기준으로 윈도우가 예측 지평 안에 들어오면 양성, 그 이전의 안정 구간이면 음성으로 " },
    { t: "환자 내부(within-patient)", b: true },
    { t: " 라벨링한다. 이 설계는 대조군이 없는 본 데이터 구조에서 필연적으로 도출되는 것이자, 동시에 개인화 관점과 자연스럽게 맞물린다(제4장)." },
  ]),
  H2("2.2 성공의 정의 — 무엇을 측정할 것인가"),
  P("평가는 임상 도입 관점을 반영하여 AUROC 단일 지표를 지양한다. 사건이 극단적으로 희귀할 때 AUROC는 오경보 문제를 구조적으로 은폐하기 때문이다. 본 과제는 다음 지표를 함께 본다."),
  TABLE(
    [2300, 3400, 3326],
    ["지표", "정의", "왜 보는가"],
    [
      ["AUPRC", "정밀도–재현율 곡선 아래 면적", "극단적 불균형에서 알람 품질을 AUROC보다 정확히 반영"],
      [[{ t: "동일 검출률에서의 알람 부담", b: true }], "검출률을 고정한 뒤 단위 윈도우당 알람 수", [{ t: "병동이 실제로 감당하는 것은 알람의 개수. 본 과제의 1급 지표", b: true }]],
      ["민감도 @ 고정 특이도", "특이도 95% 지점의 민감도", "운영 임계값 관점의 성능"],
      ["lead-time", "첫 경보 시각과 사건 시각의 간격", "조기경보의 본질 — 개입 가능 시간의 확보"],
    ],
    { align: [undefined, undefined, undefined] },
  ),
  CALLOUT([
    { t: "비교 설계의 핵심. ", b: true },
    { t: "임계값을 올리면 어떤 모델이든 알람은 줄어든다. 따라서 “알람이 줄었다”는 주장은 " },
    { t: "검출률을 동일하게 고정한 상태", b: true },
    { t: "에서만 의미를 가진다. 본 과제의 모든 알람 부담 비교는 두 점수(본 모델·NEWS)를 각각 조정해 검출률을 일치시킨 뒤 수행한다." },
  ]),
  P("요약하면, 성공의 정의는 “얼마나 잘 맞히는가”가 아니라 “같은 검출률에서 얼마나 적게, 얼마나 일찍, 왜 알리는가”이다."),
  BREAK(),
);

// ── 3. 기존 접근
children.push(
  H1("3. 기존 접근과 한계"),
  TABLE(
    [2600, 6426],
    ["접근", "한계"],
    [
      ["NEWS / MEWS (규칙 기반)", "고정 임계값으로 개인차를 반영하지 못하며 오경보가 다수 발생"],
      ["딥러닝 EWS (LSTM 등)", "연산 부담이 크고 해석이 어려움. 폐쇄망·소표본 환경에 부적합하고 병원 간 일반화가 불안정"],
      ["일반 ML 예측모델", "AUROC 위주 평가로 오경보 문제가 은폐되고, “왜 위험한지”에 대한 설명이 부재"],
      [[{ t: "상용 AI 솔루션", b: true }, { t: "\n(VUNO Med-DeepCARS 등)" }],
       [{ t: "성능 보고가 AUROC 중심 — 동일 검출률에서의 알람 부담은 공개되지 않음. 위험도 점수를 출력하나 경보 근거는 제시하지 않음. 개인 기저선을 사용하지 않는 집단 모델" }]],
    ],
  ),
  H2("3.1 선행 사례를 먼저 짚는다"),
  P("국내에는 이미 일반병동 입원환자의 심정지 위험을 감시하는 AI 의료기기가 상용화되어 있다. VUNO Med-DeepCARS는 전자의무기록에서 수집한 혈압·맥박·호흡수·체온 4종의 활력징후를 이용해 심정지 발생 위험도를 산출하며, 2021년 8월 식품의약품안전처 허가를 획득하였다."),
  P([
    { t: "이 제품의 근거 연구는 잘 정립되어 있다. 기반 알고리즘은 2개 병원 52,131명을 대상으로 한 후향 코호트에서 개발되어 기존 조기경보점수 대비 우월성이 보고되었고[2], 이후 국내 4개 수련병원에서 " },
    { t: "전향적 다기관 검증", b: true },
    { t: "이 수행되어 일반병동 환자의 심정지 및 비계획적 중환자실 이송 예측 성능이 확인되었다[3]. 즉 본 과제가 다루는 문제는 이미 임상적·규제적으로 타당성이 입증된 영역이다." },
  ]),
  CALLOUT([
    { t: "이 사실은 본 제안의 전제를 약화시키지 않는다. 오히려 강화한다. ", b: true },
    { t: "“활력징후로 심정지를 예측할 수 있는가”는 이미 임상적·규제적으로 입증된 질문이다. 따라서 본 과제가 답해야 할 질문은 그것이 아니라, " },
    { t: "“그 예측을 병동이 신뢰하고 감당할 수 있는 형태로 만들 수 있는가”", b: true },
    { t: "이다." },
  ]),
  H2("3.2 남아 있는 세 가지 간극"),
  NUM(1, [
    { t: "평가 축의 전환. ", b: true },
    { t: "선행 연구의 성능 보고는 AUROC 중심이며, 동일 검출률에서 알람이 몇 번 울리는가는 공개되지 않는다. 그러나 병동이 실제로 감당하는 것은 알람의 개수다. 극단적으로 불균형한 데이터에서 ROC 곡선이 분류기의 신뢰도를 오도할 수 있다는 것은 방법론 문헌에서 명확히 지적된 바이며, 이 경우 정밀도–재현율 평면이 더 정확한 정보를 제공한다[5]. 본 연구에서도 같은 격차를 직접 관찰하였다 — 제7-2장에서 ROC-AUC는 양쪽 모두 약 0.99로 사실상 동일한데 AUPRC는 크게 벌어진다." },
  ]),
  NUM(2, [
    { t: "경보 단위의 근거 제시. ", b: true },
    { t: "경보를 신뢰할지 판단하는 주체는 모델이 아니라 사람이다. 근거 없는 점수는 그 판단을 지원하지 못한다. 본 제안은 경보마다 기여 요인을 값과 방향으로 함께 제시한다." },
  ]),
  NUM(3, [
    { t: "개인화와 측정의 정직성. ", b: true },
    { t: "집단 모델은 개인차를 반영하지 못한다. 또한 대조군이 없는 데이터에서 오경보가 과소 추정되는 구조적 문제를 명시하지 않으면 수치가 과장된다. 본 제안은 두 문제를 모두 설계 안에서 다룬다(제5-1장)." },
  ]),
  CALLOUT([
    { t: "입력 범위의 차이. ", b: true },
    { t: "상용 솔루션이 사용하는 4종(혈압·맥박·호흡수·체온)에 더해 본 제안은 " },
    { t: "SpO₂를 포함한 6종", b: true },
    { t: "을 사용한다. 호흡성 악화 경로에서 SpO₂는 조기 신호로서 기여도가 크며, 실제로 본 연구의 SHAP 분석에서 호흡수·SpO₂의 추세가 상위 기여 요인으로 반복 확인되었다." },
  ], "FAF3E8", "B8860B"),
  H2("3.3 왜 부스팅 모델인가"),
  P("딥러닝 시계열 모델(LSTM, Transformer)은 표현력이 크지만 연산이 무겁고 해석이 어렵다. 인터넷이 차단된 안심존 환경과 수백 명 규모의 소표본에는 부적합하며, 병원 간 일반화도 불안정하다. 한편 표 형태(tabular) 데이터에서는 gradient boosting 계열이 딥러닝과 대등하거나 우수하다는 것이 대규모 벤치마크에서 반복 확인되고 있다(Grinsztajn 등, 2022)."),
  P("활력징후를 슬라이딩 윈도우 통계로 표현하면 문제는 본질적으로 tabular가 된다. 따라서 경량성, 해석가능성, 소표본 강건성이라는 세 요건을 동시에 만족하는 XGBoost가 본 과제에 적합하다. 결론적으로 본 제안은 경량·설명가능·개인화·오경보 중심이라는 빈자리를, tabular에 강한 부스팅 모델 위에 개인화와 XAI를 얹어 채운다."),
);

// ── 4. 제안 아이디어
children.push(
  H1("4. 제안 아이디어"),
  P("두 축의 차별화와, 데이터의 약점을 방법론으로 전환하는 세 번째 축으로 구성된다."),
  H2("4.1 개인화 기저선 이탈 (Personalized Baseline Deviation)"),
  P("병동 공통 임계값이 아니라, 각 환자의 초기 안정기를 개인 기저선으로 삼아 그로부터의 이탈을 위험신호로 사용한다. 구체적으로 각 활력징후에 대해 개인 기저선 대비 최근값 편차와 평균 편차를 피처로 생성한다."),
  P([
    { t: "병동 전체 기준으로는 정상 범위에 속하는 값이라도 특정 환자에게는 큰 이탈일 수 있다. 반대로 평소 수치가 높은 환자의 경우, 고정 임계값 기준으로는 상시 경보 대상이지만 개인 기저선 기준으로는 안정 상태다. " },
    { t: "환자가 곧 자기 자신의 대조군이 되는 것", b: true },
    { t: "이 이 설계의 핵심이며, 이는 개인차에서 비롯되는 오경보를 줄이고 병원 간 전이성을 높인다." },
  ]),
  H2("4.2 설명가능성(XAI) 내장"),
  P("모든 경보에 대해 왜 위험한지를 함께 제시한다. SHAP 기반으로 해당 시점의 상위 기여 활력징후와 그 방향(상승/하강), 추세를 산출하여 경보와 동시에 표시한다. 이는 세 가지 실질적 효과를 갖는다."),
  BULLET("임상 신뢰와 수용성 — 근거를 검증할 수 있는 경보만이 실제 의사결정에 반영된다"),
  BULLET("간호 개입 우선순위 판단 — 어떤 계통의 악화인지에 따라 대응이 달라진다"),
  BULLET("모델 검증 — 개발자가 모델의 오작동을 조기에 발견할 수 있다"),
  H2("4.3 case-only 데이터의 역설을 정면으로 다룬다"),
  P([
    { t: "경북대 데이터는 심정지가 발생한 환자만으로 구성되어 대조군이 없다. 본 제안은 이를 한계로만 두지 않고 " },
    { t: "within-patient 설계의 근거", b: true },
    { t: "로 전환한다 — 안정기를 음성, 심정지 직전 구간을 양성으로 하여 환자 내부에서 라벨을 부여한다. 동시에 이 구조가 만들어내는 측정의 한계를 명시하고 보정 경로를 설계에 포함한다(제5-1장)." },
  ]),
  H2("4.4 보조 참신성 — 심정지 표현형 발견"),
  P("심정지 직전의 활력징후 궤적을 비지도 군집화하면 서로 다른 악화 아형이 드러난다. 개발 데이터에서는 호흡성 악화형, 순환성 악화형, 혼합형의 세 아형이 자동으로 분리되었다. 이는 두 가지 확장을 가능하게 한다 — 표현형별 맞춤 경보 임계값 설정, 그리고 악화 경로에 따른 대응 프로토콜 차별화다."),
  CALLOUT([
    { t: "한 문장 차별점. ", b: true },
    { t: "정확도 경쟁을 우회하고, 개인화·설명·오경보라는 임상 도입의 실제 장벽을 푼다.", b: true },
  ]),
);

// ── 5. 데이터 활용
children.push(
  H1("5. 데이터 활용 방안"),
  P("경북대학교병원 입원환자 활력징후 데이터는 환자 기본정보(KHTH_PINFO)와 시점별 활력징후(KHTH_VITAL) 두 테이블로 구성된다. 2023~2025년 입원 중 심정지가 발생한 20~80세 573명을 대상으로 하며, 전원 심정지 후 사망한 환자다. 활력징후는 수 분 간격으로 불규칙하게 기록되고, 입원 24시간 이내 심정지 사례는 제외되어 모든 환자가 최소 24시간 이상의 관찰 구간을 갖는다."),
  H2("5.1 테이블 활용 설계"),
  TABLE(
    [2100, 6926],
    ["항목", "활용 방안"],
    [
      ["테이블 조인", "KHTH_PINFO(연령대·성별·입퇴원일·심정지시각 CARDT·사망시각)와 KHTH_VITAL(시점별 HR/SBP/DBP/BT/SPO2/RR)을 PATID + INDD로 연결"],
      ["라벨링", "CARDT 기준 정확한 심정지 시각을 사용해 within-patient 양성/음성 부여"],
      ["관찰창 확보", "대회 제외기준(입원 24시간 이내 심정지 제외)에 따라 전 환자 24시간 이상의 활력징후 구간이 보장됨"],
      ["정제", "자유문자열 VS_RSLT의 센서 아티팩트·단위 이상값 처리. 생리학적 범위를 벗어난 값은 결측으로 전환(개발 과정에서 MIMIC 실데이터로 확인된 이슈를 사전 방어)"],
      ["정적 변수", "연령대·성별을 보조 피처로 활용하여 예후 보정 및 설명 강화"],
    ],
  ),
  H2("5.2 case-only 구조에서 오경보를 어떻게 측정할 것인가"),
  P("본 데이터는 573명 전원이 심정지 환자다. 오경보 저감이 본 제안의 논지인 이상, 이 구조가 무엇을 가능하게 하고 무엇을 불가능하게 하는지를 분명히 해 두는 것이 실현성의 전제다."),
  H3("측정 가능한 것 — 환자 내부 오경보"),
  P("각 환자의 시간축에는 안정 구간이 존재한다. 심정지 수 시간 전의 안정 구간에서 울린 경보는 예측 지평 기준으로 명백한 오경보이며, 따라서 환자 내부 오경보율과 알람 부담은 그대로 산출된다. within-patient 설계가 수행하는 역할이 이것이다."),
  H3("측정 불가능한 것 — 병동 수준의 알람 부담"),
  P([
    { t: "원내 심정지는 입원 건수 대비 극히 드문 사건이므로, 실제 병동에서 발생하는 알람의 대부분은 끝내 심정지에 이르지 않는 환자에게서 나온다. 그런데 그 집단이 본 데이터에는 존재하지 않는다. 즉 " },
    { t: "case-only 데이터에서 산출한 오경보 수치는 양성이 인위적으로 농축된 표본에서 나온 값이므로 실제 병동보다 낙관적으로 추정된다", b: true, c: WARN },
    { t: ". 이 점을 명시하지 않은 채 알람 감소폭을 제시하는 것은 과장이다." },
  ]),
  H3("보정 경로"),
  P("본 제안은 경북대 데이터 단독으로 within-patient 오경보와 알람 부담을 산출하는 것을 기본 설계로 하며, 병동 수준 추정치는 다음 두 경로로 보완한다."),
  NUM(1, [
    { t: "외부 대조군 사전학습 후 모델 반입 (주 경로). ", b: true },
    { t: "심정지 미발생 환자를 포함한 공개 데이터(MIMIC-IV)로 안심존 외부에서 모델을 사전학습하고, 학습된 모델을 반입 승인 절차를 거쳐 안심존에 들여와 경북대 데이터로 추가 학습·보정한다. 대조군에서 학습한 오경보 특성을 case-only 데이터에 이식하는 설계다. 운영기관 확인 결과 외부 데이터·학습 모델·가중치 파일 모두 사전 신청 후 승인 시 반입이 가능하다." },
  ]),
  NUM(2, [
    { t: "타 기관 제공 데이터 결합 (보조 경로). ", b: true },
    { t: "대회 제공 타 기관 데이터에 심정지 미발생 환자군이 포함될 경우 외부 대조군으로 활용한다. 안심존 운영 측에는 결합에 대한 제한이 없음을 확인하였다." },
  ]),
  CALLOUT([
    { t: "두 경로가 모두 확보되지 않더라도 본 설계는 성립한다. ", b: true },
    { t: "그 경우 보고되는 알람 부담은 “심정지 발생 환자군 내부 기준”임을 지표명에 명시하고, 병동 수준으로의 일반화는 후속 과제로 남긴다." },
  ]),
  P([
    { t: "데이터 이용 조건. ", b: true },
    { t: "MIMIC-IV는 제3자 공유를 금지하는 데이터 이용 협약(DUA) 하에 제공된다. 따라서 원자료 자체의 반입은 수행하지 않고, 파생 산출물인 학습 모델만 반입하는 방식을 택한다. 원자료 접근에 필요한 PhysioNet CITI 인증은 취득 완료하였다." },
  ], { size: 19, color: MUTED }),
  H2("5.3 데이터 품질 점검 및 정제 규칙"),
  P("활력징후 원자료에는 센서 아티팩트, 단위 혼재, 기록 누락이 포함된다. 이를 사전에 정의된 규칙으로 처리하지 않으면 모델이 기계의 오류를 환자의 악화로 학습한다. 아래 규칙은 개발 과정에서 MIMIC-IV 실데이터를 다루며 실제로 확인된 문제에 대응해 수립한 것이며, 코드로 구현되어 있다."),
  TABLE(
    [1900, 2500, 4626],
    ["점검 항목", "판정 기준", "처리"],
    [
      ["생리학적 범위 이탈", "각 활력징후별 물리적으로 불가능한 값", "결측으로 전환 후 대치 (삭제하지 않음 — 행 삭제는 시간축을 훼손)"],
      ["체온 단위 혼재", "화씨로 기록된 값이 섞여 있는 경우", "섭씨로 자동 변환"],
      ["자유문자열 결과값", "KHTH_VITAL의 VS_RSLT가 문자열", "수치 파싱 실패 시 결측 처리"],
      ["기록 밀도 부족", "윈도우 내 유효 관측 비율 50% 미만", "해당 윈도우 제외 (min_valid_fraction = 0.5)"],
      ["시각 정합성", "PATID+INDD 조인 후 CARDT와 활력징후 시각의 정합", "불일치 환자 별도 확인 후 판단"],
    ],
  ),
  CALLOUT([
    { t: "삭제가 아니라 결측 처리인 이유. ", b: true },
    { t: "이상값이 포함된 시점을 통째로 삭제하면 시계열에 구멍이 생겨 추세(slope)와 변화량 피처가 왜곡된다. 값만 결측으로 바꾸고 학습 집합의 중앙값으로 대치하면 시간축은 보존된다." },
  ]),
  BREAK(),
);

// ── 6. 분석 방법론
children.push(
  H1("6. 분석 방법론"),
  H2("6.1 파이프라인 개요"),
  TABLE(
    [1000, 8026],
    ["단계", "내용"],
    [
      ["1", "활력징후 시계열 적재 및 정제 (센서 아티팩트·단위 이상값 처리)"],
      ["2", "슬라이딩 윈도우 통계 피처 생성 — 활력징후별 평균·표준편차·최소·최대·최근값·추세(slope)·변화량 및 shock index"],
      ["3", [{ t: "개인 기저선 이탈 피처 생성", b: true }, { t: " — {vital}_last_dev, {vital}_mean_dev" }]],
      ["4", "within-patient 라벨링 — 예측 지평 내 심정지 발생 여부"],
      ["5", [{ t: "cost-sensitive XGBoost 학습", b: true }, { t: " (비교군: NEWS 임상 표준)" }]],
      ["6", "평가 — AUPRC·AUROC·민감도@95%특이도·오경보율·알람 부담·lead-time"],
      ["7", "설명 — SHAP 전역 기여도 및 경보별 상위 기여 요인"],
    ],
    { align: [AlignmentType.CENTER, undefined] },
  ),
  H2("6.2 피처 설계"),
  P("각 윈도우는 활력징후별로 수준(평균·최근값), 변동성(표준편차·최소·최대), 추세(slope), 변화량을 담는다. 이는 “지금 값이 얼마인가”뿐 아니라 “어느 방향으로 얼마나 빠르게 움직이는가”를 포착하기 위함이다. 실제로 본 연구의 SHAP 분석에서 상위 기여 요인으로 반복 확인된 것은 절대값이 아니라 호흡수·SpO₂·맥박의 추세였다."),
  P("여기에 핵심 차별 피처인 개인 기저선 이탈을 더한다. 각 환자의 초기 안정기 평균을 기저선으로 삼아, 최근값과 윈도우 평균의 편차를 각각 산출한다."),
  H2("6.3 학습 설계와 누수 방지"),
  BULLET([{ t: "환자 단위 분할. ", b: true }, { t: "같은 환자의 윈도우가 학습·평가 집합에 나뉘어 들어가면 모델이 해당 환자의 패턴을 기억한 상태로 평가받게 된다. 시계열 의료 데이터에서 가장 흔한 누수 유형이므로 분할은 반드시 환자 단위로 수행한다." }]),
  BULLET([{ t: "결측 대치. ", b: true }, { t: "학습 집합의 중앙값으로 대치하여 평가 집합의 정보가 학습에 유입되지 않도록 한다." }]),
  BULLET([{ t: "불균형 처리. ", b: true }, { t: "scale_pos_weight를 음성/양성 비율로 설정하고, 하이퍼파라미터 탐색은 ROC-AUC가 아니라 AUPRC를 최대화하는 방향으로 수행한다." }]),
  BULLET([{ t: "재현성. ", b: true }, { t: "분할 시드를 고정하고, 서로 다른 시드에서 결론의 방향이 유지되는지를 반복 확인한다." }]),
  H2("6.4 피처 명세와 윈도우 파라미터"),
  P("아래는 구현된 파이프라인의 실제 명세다. 제안 단계의 개념이 아니라 코드에 반영되어 동작 중인 값이다."),
  TABLE(
    [2400, 1500, 5126],
    ["구성", "개수", "내용"],
    [
      ["활력징후", "6종", "맥박(pulse), 수축기혈압(sbp), 이완기혈압(dbp), 체온(temperature), SpO₂(spo2), 호흡수(resp_rate)"],
      ["윈도우 통계", "7종", "평균(mean), 표준편차(std), 최소(min), 최대(max), 최근값(last), 추세(slope), 변화량(delta)"],
      ["기본 피처", [{ t: "44", b: true }], "6종 × 7통계 = 42, 여기에 shock index 관련 2개"],
      [[{ t: "개인 기저선 이탈", b: true }], [{ t: "12", b: true }], [{ t: "{vital}_last_dev, {vital}_mean_dev — 6종 × 2 (본 제안의 핵심 차별 피처)", b: true }]],
      ["정적 피처", "2", "연령대(age), 성별(sex)"],
      [[{ t: "합계", b: true }], [{ t: "58", b: true }], [{ t: "윈도우 1개당 입력 차원", b: true }]],
    ],
    { align: [undefined, AlignmentType.CENTER, undefined],
      headerAlign: [undefined, AlignmentType.CENTER, undefined] },
  ),
  P("표 3. 시간 파라미터", { size: 19, bold: true, before: 200, after: 100 }),
  TABLE(
    [2600, 1400, 5026],
    ["파라미터", "기본값", "의미"],
    [
      ["관찰 윈도우", "8시간", "각 윈도우가 참조하는 과거 구간"],
      ["예측 지평", "1시간", "윈도우 종료 시점 이후 심정지 발생을 양성으로 판정하는 구간"],
      ["gap", "0시간", "관찰 종료와 예측 시작 사이의 공백 (운영 시 지연을 반영하려면 확대)"],
      ["슬라이딩 간격", "1시간", "윈도우 생성 간격"],
      ["개인 기저선 구간", "6시간", "각 환자의 초기 안정기 — 개인 기저선 산출 기준"],
      ["유효 관측 최소 비율", "50%", "이 비율 미만인 윈도우는 학습·평가에서 제외"],
    ],
    { align: [undefined, AlignmentType.CENTER, undefined],
      headerAlign: [undefined, AlignmentType.CENTER, undefined] },
  ),
  CALLOUT([
    { t: "예측 지평은 조정 가능한 설계 변수다. ", b: true },
    { t: "지평을 늘리면 lead-time이 길어지는 대신 양성 정의가 느슨해져 알람이 늘어난다. 본선에서는 경북대 데이터의 실제 기록 밀도를 확인한 뒤 임상적으로 유효한 개입 시간(신속대응팀 도착 소요 등)을 기준으로 지평을 확정한다." },
  ]),
  H2("6.5 폐쇄망 적합 경량 스택"),
  P("안심존은 인터넷이 차단된 오프라인 환경이며 사전신고한 패키지만 사용할 수 있다. 이에 따라 의존성을 numpy, pandas, scikit-learn, xgboost, shap, matplotlib으로 최소화하였다. 외부 다운로드나 외부 API 호출에 의존하는 구성요소는 설계에 포함하지 않는다."),
  CALLOUT([
    { t: "이 제약은 모델 선택의 실질적 근거이기도 하다. ", b: true },
    { t: "딥러닝 프레임워크를 배제하고 경량 부스팅 모델을 선택한 것은 성능 판단만이 아니라, 폐쇄망에서 실제로 실행 가능한 구성을 우선한 결과다." },
  ]),
  BREAK(),
);

// ── 7. 검증 결과
children.push(
  H1("7. 검증 결과"),
  P("수치의 출처를 두 갈래로 나누어 제시한다. 제7-1장은 공개 실데이터에서 얻은 실측치로, 파이프라인이 실제 임상 데이터에서 끝까지 동작하며 핵심 주장이 재현됨을 보인다. 제7-2장은 심정지 라벨 체계에서의 파이프라인 작동 시연으로, 합성·개발 데이터 기반이며 실제 환자 성능이 아니다."),
  H2("7.1 공개 실데이터 검증 — PhysioNet/CinC Challenge 2019"),
  P("동일 파이프라인(정제 → 슬라이딩 윈도우 → 개인 기저선 이탈 피처 → cost-sensitive XGBoost → SHAP)을 공개 실데이터에 적용해 end-to-end 동작과 핵심 주장을 검증하였다."),
  CALLOUT([
    { t: "대상 사건은 패혈증 발병이며 심정지가 아니다. ", b: true, c: WARN },
    { t: "본 결과는 파이프라인의 실현 가능성에 대한 근거이지, 심정지 예측 성능의 예측치가 아니다." },
  ], "FBF2F2", WARN),
  BULLET("규모: 20,336명 적재 → 환자 단위 분할 train 15,886명 / test 3,972명(126,558 윈도우), 양성률 1.19%"),
  BULLET("설정: 입력 창 과거 8시간, 예측 지평 향후 6시간"),
  P("표 1. 동일 검출률에서의 알람 부담 (test 3,972명 전수 기준)", { size: 19, bold: true, before: 160, after: 100 }),
  TABLE(
    [2200, 2400, 2200, 2226],
    ["검출률(민감도)", "본 모델 알람/100", "NEWS 알람/100", "알람 감소"],
    [
      [[{ t: "50%", b: true }], [{ t: "24.8", b: true }, { t: " (특이도 0.755)" }], "41.3 (0.589)", [{ t: "40%", b: true, c: ACCENT }]],
      ["70%", [{ t: "45.8", b: true }, { t: " (0.545)" }], "59.2 (0.410)", "23%"],
      ["90%", "76.3 (0.238)", "100.0 (0.000)", [{ t: "비교 불성립", i: true, c: WARN }]],
    ],
    { align: [AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER],
      headerAlign: [AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER] },
  ),
  P("AUPRC 0.027(무작위 기준선 0.0119 대비 2.3배), ROC-AUC 0.679, lead-time 중앙값 33시간.", { before: 140 }),
  BULLET([{ t: "핵심 주장의 재현. ", b: true }, { t: "검출률을 고정한 상태에서 알람 수가 실제로 감소한다. 이 비교는 임계값을 올려 알람을 줄이는 것과 무관하다 — 검출률이 동일한 지점끼리만 비교하였다." }]),
  BULLET([{ t: "재현성. ", b: true }, { t: "서로 다른 환자 분할 시드 3회 반복에서 AUPRC 우위와 알람 감소가 모두 같은 방향으로 재현되었다(별도 실행이므로 수치 자체는 위 표와 다르다)." }]),
  BULLET([{ t: "AUPRC 0.027의 해석. ", b: true }, { t: "절대값은 낮으나, 양성률 1.19%인 문제에서 무작위 분류기의 AUPRC가 0.0119이므로 기준선 대비로 읽어야 하는 지표다." }]),
  BULLET([{ t: "산출물. ", b: true }, { t: "판단 근거를 사전계산해 정적 스냅샷으로 서빙하는 대시보드까지 구현·배포하였다. 경보마다 기여 요인 상위 3개를 값과 방향으로 제시한다." }]),
  BREAK(),
);

children.push(
  H2("7.2 심정지 파이프라인 작동 시연"),
  CALLOUT([
    { t: "아래는 심정지 라벨 체계에서 파이프라인이 구동됨을 보이는 시연이다. 합성 데이터 기반이므로 실제 환자 성능으로 인용해서는 안 된다. ", b: true, c: WARN },
    { t: "본선에서 경북대 데이터로 동일 파이프라인을 적용하여 실측치로 대체한다." },
  ], "FBF2F2", WARN),
  TABLE(
    [3226, 2900, 2900],
    ["지표", "XGBoost", "NEWS"],
    [
      ["AUPRC", [{ t: "0.84", b: true }], "0.63"],
      ["ROC-AUC", "약 0.99", "약 0.99"],
      ["알람 수 / 100 윈도우 @ 90% 민감도", [{ t: "2.0", b: true }], "2.9"],
      ["lead-time (중앙값)", "심정지 약 3시간 전", "—"],
    ],
    { align: [undefined, AlignmentType.CENTER, AlignmentType.CENTER],
      headerAlign: [undefined, AlignmentType.CENTER, AlignmentType.CENTER] },
  ),
  P("위 수치는 본 제안서의 그림 1~4를 생성한 실행 기준이다. 합성 코호트를 재생성하면 AUPRC는 XGBoost 0.77~0.97, NEWS 0.42~0.63 범위에서 변동한다.", { size: 18, color: MUTED, before: 100, after: 120 }),
  P([
    { t: "주목할 점은 ROC-AUC가 양쪽 모두 약 0.99로 사실상 동일한데 AUPRC는 크게 벌어진다는 것이다. " },
    { t: "즉 ROC-AUC만 보면 두 방법이 동등해 보이지만, 실제 알람 품질에는 상당한 격차가 존재한다.", b: true },
    { t: " 이는 제1장과 제3장에서 제기한 “AUROC가 오경보 문제를 은폐한다”는 논지를 직접 뒷받침한다." },
  ]),
  ...FIG("vitals_pr_curve.png", 0.855, "그림 1. PR 곡선 비교 — ROC로는 보이지 않는 격차가 정밀도–재현율 평면에서 드러난다", 360),
  ...FIG("vitals_alarm_burden.png", 0.712, "그림 2. 동일 검출률에서의 알람 부담 — 고민감도 구간에서 NEWS의 알람이 급증한다", 420),
  ...FIG("vitals_trajectory.png", 0.544, "그림 3. 환자 악화 궤적과 경보 시점 — 위험도 상승이 사건에 선행한다", 510),
  ...FIG("vitals_lead_time.png", 0.711, "그림 4. lead-time 분포 — 개입 가능 시간의 확보 정도", 415),
);

children.push(
  BREAK(),
  H3("심정지 표현형 발견 (보조 참신성)"),
  P("심정지 직전의 활력징후 궤적을 비지도 군집화하면 서로 다른 악화 아형이 자동으로 분리된다. 아래는 개발 데이터 296명을 3개 군집으로 나눈 결과이며, 각 값은 개인 기저선 대비 평균 변화량이다."),
  TABLE(
    [1500, 1100, 1100, 1100, 1300, 1100, 1826],
    ["아형", "맥박", "수축기", "이완기", "SpO₂", "호흡수", "해석"],
    [
      [[{ t: "A (86명)", b: true }], "+9.2", "−8.2", "−4.4", [{ t: "−8.0", b: true, c: WARN }], [{ t: "+9.0", b: true, c: WARN }], "호흡성 악화형 — SpO₂ 저하와 호흡수 상승이 지배적"],
      [[{ t: "B (132명)", b: true }], "+12.7", "−11.0", "−6.1", "−3.3", "+3.7", "경증 혼합형 — 모든 계통이 완만하게 변화"],
      [[{ t: "C (78명)", b: true }], [{ t: "+22.3", b: true, c: WARN }], [{ t: "−20.8", b: true, c: WARN }], [{ t: "−12.0", b: true, c: WARN }], "−3.9", "+4.5", "순환성 악화형 — 빈맥과 혈압 저하가 지배적"],
    ],
    { align: [undefined, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, undefined],
      headerAlign: [undefined, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, undefined] },
  ),
  P("아형 A와 C는 지배적인 악화 계통이 다르므로 감시해야 할 활력징후와 개입 방식도 달라진다. 이는 단일 임계값 체계로는 포착되지 않는 구조이며, 표현형별 경보 임계값 차등화와 대응 프로토콜 분리로 확장할 수 있다.", { before: 140 }),
  ...FIG("vitals_phenotypes.png", 0.442, "그림 5. 심정지 표현형 군집 — 기저선 대비 변화량 기준 아형 분리", 545),
  H2("7.3 인용 시 반드시 함께 밝히는 조건"),
  P("실데이터 검증 과정에서 함께 확인된 한계다. 숨기지 않고 방법론의 일부로 다룬다."),
  NUM(1, [
    { t: "“알람 40% 감소”는 조건부 수치다. ", b: true },
    { t: "전체 데이터와 검출률 50% 지점에서 성립하며, 4,000명 부분집합에서는 10%로 떨어진다. 데이터 규모에 대한 의존성이 크다." },
  ]),
  NUM(2, [
    { t: "개인 기저선 이탈 피처는 주된 기여 요인이 아니었다. ", b: true },
    { t: "실데이터 전역 기여도에서 4위·6위이며, 1~3위는 호흡수·체온의 절대값이다. 합성 데이터에서 관찰된 AUPRC 0.76 → 0.84 상승은 실데이터에서 그대로 재현되지 않았다. “기여한다”까지가 정확한 표현이며 “핵심 요인”은 과장이다." },
  ]),
  NUM(3, [
    { t: "검출률 90% 지점은 비교에서 제외한다. ", b: true },
    { t: "해당 지점의 NEWS 특이도가 0이 되어 사실상 전원 알람 상태가 되므로 비교가 성립하지 않는다." },
  ]),
  CALLOUT([
    { t: "한계를 먼저 밝히는 것이 본 제안의 방법론적 태도다. ", b: true },
    { t: "특히 두 번째 항목은 본 제안의 핵심 차별점에 대한 불리한 결과임에도 명시한다. 실데이터에서 검증하지 않았다면 발견할 수 없었을 사실이며, 이를 확인하고 보고하는 것 자체가 본선 수행의 신뢰성을 담보한다." },
  ]),
  BREAK(),
);

// ── 7.4 설명가능성 실증
children.push(
  H2("7.4 설명가능성 실증 — 근거가 화면에 도달하는가"),
  P("설명가능성은 개념으로 주장할 것이 아니라 실제로 임상의가 읽을 수 있는 형태까지 도달해야 한다. 본 팀은 제7-1장의 공개 실데이터 검증 결과를 대상으로, 경보의 근거를 사전계산해 화면에 제시하는 시스템을 구현·배포하였다."),
  ...FIG("dashboard_screenshot.png", 0.656, "그림 6. 구현·배포된 조기경보 화면 — 위험도 추이, 활력징후, 기여 요인, 알람 부담을 한 화면에 제시", 545),
  P("화면이 제공하는 정보는 다음과 같다."),
  BULLET([{ t: "위험도 추이와 경보 시점. ", b: true }, { t: "시점별 위험도와 임상 표준(NEWS)을 같은 시간축에 나란히 두되, 축을 공유하지 않는 별도 플롯으로 그려 없는 상관을 시사하지 않는다." }]),
  BULLET([{ t: "활력징후 6종의 궤적. ", b: true }, { t: "개인 기저선을 가로선으로 겹쳐 표시하여, 병동 기준으로는 정상이지만 그 환자에게는 이탈인 상태를 눈으로 확인할 수 있다." }]),
  BULLET([{ t: "경보의 기여 요인. ", b: true }, { t: "각 경보 시점에서 상위 3개 기여 요인을 값과 방향(상승/하강)으로 제시한다. “왜 울렸는가”가 화면에 있다." }]),
  BULLET([{ t: "임계값 조정과 그 대가. ", b: true }, { t: "임계값은 학습 이후의 운영 결정이므로, 재학습 없이 조정하며 검출률과 알람 부담이 어떻게 맞바꾸어지는지를 즉시 확인할 수 있다." }]),
  H3("모델과 무관한 두 번째 안전 채널"),
  P([
    { t: "검증 과정에서 예측 라벨 정의의 구조적 한계를 발견하였다. 이미 상태가 심각하게 무너진 환자는 사건 시점을 지났거나 다른 원인으로 악화 중인 경우가 많아, “향후 N시간 내 발생 확률”이 오히려 낮게 산출된다. " },
    { t: "모델은 학습한 질문에 정확히 답한 것이지만, 그 낮은 점수가 화면에서 “안정”으로 읽히면 위험하다.", b: true },
  ]),
  P("이에 대응하여 모델 점수와 무관하게 활력징후만을 직접 판독하는 별도 채널을 두고, 생리학적으로 붕괴 수준인데 모델이 조용한 경우 화면 상단에 경고를 표시하도록 하였다. 모델을 억지로 수정하는 대신 시스템이 자신의 한계를 인지하게 만든 것이며, 본선의 심정지 라벨 체계에도 동일하게 적용한다."),
  CALLOUT([
    { t: "이 발견은 실데이터를 끝까지 다루지 않으면 나오지 않는다. ", b: true },
    { t: "제안 단계의 설계 검토로는 드러나지 않는 문제이며, 본 팀이 파이프라인을 실데이터에서 화면까지 완주했기 때문에 확인할 수 있었다. 본선에서 경북대 데이터를 다룰 때에도 같은 방식의 점검을 수행한다." },
  ]),
);

// ── 8. 기대효과
children.push(
  H1("8. 기대효과 및 파급성"),
  P("본 제안의 가치는 정확도 지표의 개선을 넘어 “실제로 쓰이는 조기경보”를 만드는 데 있다. 오경보를 줄여 alarm fatigue를 완화하면, 그동안 성능이 우수해도 무시되던 경보가 비로소 임상 의사결정에 반영되어 심정지 조기대응과 생존율 개선으로 이어질 수 있다."),
  H2("8.1 기대효과"),
  TABLE(
    [2100, 6926],
    ["영역", "기대효과"],
    [
      [[{ t: "환자안전", b: true }], "오경보를 줄여 실제 병동에서 사용되는 조기경보 체계를 구현. 심정지 조기대응과 생존율 개선에 기여"],
      [[{ t: "의료 현장", b: true }], "설명가능 경보를 통해 간호 인력 배치와 중환자 조기 스크리닝 프로토콜을 고도화. 어떤 계통의 악화인지가 경보와 함께 제시되므로 대응의 우선순위 판단이 가능"],
      [[{ t: "확장성", b: true }], "개인 기저선 기반이므로 타 병원 이식이 용이. 활력징후만으로 동작하여 추가 장비·검사 없이 도입 가능 — 도입 장벽이 낮음"],
      [[{ t: "데이터 활용", b: true }], "case-only 구조의 미개방 데이터를 within-patient 설계로 활용하는 방법론을 제시. 유사 구조의 다른 미개방 데이터셋에 재사용 가능"],
    ],
  ),
  H2("8.2 발전 로드맵"),
  BULLET([{ t: "단기. ", b: true }, { t: "경북대 데이터에서의 실측 성능 확보 및 표현형별 경보 전략 수립" }]),
  BULLET([{ t: "중기. ", b: true }, { t: "다기관 외부검증 — 개인화 설계의 병원 간 전이성을 정량적으로 확인" }]),
  BULLET([{ t: "장기. ", b: true }, { t: "ECG·RR interval·HRV 등 고해상도 신호로 입력을 확장하여 조기 포착 시점을 앞당김" }]),
  CALLOUT([
    { t: "본 제안이 남기는 것은 모델 하나가 아니라 평가 프레임이다. ", b: true },
    { t: "“동일 검출률에서의 알람 부담”이라는 지표를 조기경보 평가의 표준 항목으로 제시함으로써, 후속 연구와 제품이 오경보 문제를 은폐하지 않고 비교될 수 있는 기준을 만든다." },
  ]),
  BREAK(),
);

// ── 9. 추진 계획
children.push(
  H1("9. 추진 계획 및 안심존 활용"),
  P("안심존은 오프라인 폐쇄망이며 반입과 반출에 각각 승인 절차가 있다. 운영기관 확인을 거쳐 아래 제약을 전제로 계획을 수립하였다."),
  H2("9.1 확인된 운영 조건"),
  TABLE(
    [4200, 4826],
    ["항목", "조건"],
    [
      ["사전신고 → 이용 개시", "약 1주 소요"],
      ["외부 공개 데이터 반입", "신청 및 승인 시 가능"],
      ["외부 학습 모델·가중치 파일 반입", "신청 및 승인 시 가능"],
      ["분석 결과 반출", "보고서용 수치·그림 등은 반출 가능"],
      [[{ t: "모델 반출", b: true }], [{ t: "불가를 전제로 계획", b: true, c: WARN }]],
    ],
  ),
  H2("9.2 이 제약이 설계에 미치는 영향"),
  P([
    { t: "안심존 내부에서 경북대 데이터로 학습한 모델은 외부로 반출되지 않는다. 따라서 산출물을 두 갈래로 분리한다. " },
    { t: "경북대 데이터 기반 결과는 수치·그림·보고서 형태로 반출", b: true },
    { t: "하고, 동작 시연이 필요한 경우에는 공개 데이터로 학습한 모델을 사용한다. 최종 성과물이 반출 불가 자산에 의존하지 않도록 하는 것이 목적이다." },
  ]),
  H2("9.3 작업 배치 — 밖에서 확정하고, 안에서 실행한다"),
  P("폐쇄망 내부에서의 시행착오는 비용이 크다. 따라서 설계 결정은 모두 외부에서 마치고, 안심존에서는 확정된 설계를 실행·검증하는 데 집중한다."),
  TABLE(
    [2100, 6926],
    ["위치", "수행 내용"],
    [
      ["안심존 외부", "공개 데이터로 피처 설계·하이퍼파라미터·임계값 정책 확정. 대조군 기반 사전학습. 코드 및 테스트 완성"],
      ["반입", "확정된 코드 + 사전학습 모델 (승인 절차)"],
      ["안심존 내부", "경북대 데이터로 추가 학습·보정. 지표 산출 및 그림 생성"],
      ["반출", "수치 · 그림 · 보고서"],
    ],
  ),
  H2("9.4 일정"),
  TABLE(
    [2400, 6626],
    ["시점", "내용"],
    [
      ["예선 기간", "안심존 방문 완료 — 데이터 구조 및 값 범위 확인 (수행 완료)"],
      ["본선 진출 확정 직후", "안심존 사전신고 즉시 신청 — 이용 개시까지 약 1주가 소요되므로 임계경로"],
      ["본선 초기", "사전학습 모델 및 코드 반입 신청 · 승인"],
      ["본선 중기", "경북대 데이터 학습·평가·설명 산출"],
      ["본선 후기", "결과물 반출 신청 — 심의 기간을 감안하여 최종 제출 기준 여유를 두고 조기 신청"],
    ],
  ),
  H2("9.5 폐쇄망 대응"),
  P("사전신고 패키지는 numpy, pandas, scikit-learn, xgboost, shap, matplotlib으로 최소화한다. 외부 다운로드에 의존하는 구성요소는 두지 않으며, 외부 API가 필요한 기능(자연어 생성 등)은 안심존 대상에서 제외하고 결정론적으로 계산되는 근거 제시만을 사용한다."),
  H2("9.6 위험 요인 및 대응 계획"),
  P("본선 수행에서 발생 가능한 위험과 그 대응을 사전에 정의하였다. 각 항목은 발생 시 계획을 수정하는 것이 아니라, 이미 준비된 대안으로 전환하는 방식으로 처리한다."),
  TABLE(
    [2500, 1200, 5326],
    ["위험 요인", "영향", "대응"],
    [
      ["표본 부족 — 573명은 딥러닝은 물론 복잡한 모델에도 작은 규모", [{ t: "높음", b: true, c: WARN }], "경량 부스팅 모델 채택이 이미 이에 대한 대응. 환자 단위 교차검증으로 분산을 함께 보고하고, 단일 시드 결과를 성능으로 주장하지 않음"],
      ["대조군 부재로 인한 오경보 과소 추정", [{ t: "높음", b: true, c: WARN }], "제5-2장의 두 경로(외부 대조군 사전학습 후 모델 반입, 타 기관 데이터 결합). 두 경로 모두 불가 시 지표명에 “환자군 내부 기준”을 명시"],
      ["기록 밀도 불균일 — 활력징후가 수 분 간격으로 불규칙 기록", "중간", "유효 관측 비율 50% 기준으로 윈도우를 선별. 밀도가 낮은 구간은 학습·평가에서 제외하고 그 비율을 함께 보고"],
      ["안심존 이용 개시 지연", "중간", "본선 진출 확정 즉시 사전신고 — 약 1주 소요를 일정에 반영 완료. 대기 기간에는 외부에서 설계 확정 작업 수행"],
      ["반입 승인 지연 또는 불허", "중간", "사전학습 모델 없이 안심존 내부 학습만으로도 성립하는 설계. 반입은 성능 향상 수단이지 전제 조건이 아님"],
      ["반출 심의 지연", "낮음", "최종 제출 기준 여유를 두고 조기 신청. 반출 대상은 수치·그림으로 한정해 심의 부담 최소화"],
      ["라벨 정의의 구조적 한계 (제7-4장)", "중간", "모델과 무관한 생리학적 안전 채널을 병행 운영. 극단 상태에서의 점수 거동을 별도 점검 항목으로 관리"],
    ],
    { align: [undefined, AlignmentType.CENTER, undefined],
      headerAlign: [undefined, AlignmentType.CENTER, undefined] },
  ),
  BREAK(),
);

// ── 10. 윤리 · 보안
children.push(
  H1("10. 윤리 · 개인정보 · 데이터 보안"),
  P("의료데이터를 다루는 과제인 만큼, 성능과 무관하게 반드시 지켜야 할 조건을 설계에 포함한다."),
  H2("10.1 데이터 취급 원칙"),
  BULLET([{ t: "안심존 외부 반출 금지. ", b: true }, { t: "경북대학교병원 데이터는 어떤 형태로도 안심존 외부로 반출하지 않는다. 반출 대상은 승인된 분석 결과(수치·그림)로 한정한다." }]),
  BULLET([{ t: "재식별 시도 금지. ", b: true }, { t: "비식별 처리된 데이터에 대해 재식별을 시도하지 않으며, 개별 환자를 특정할 수 있는 형태의 결과는 산출·보고하지 않는다." }]),
  BULLET([{ t: "사례 제시의 익명성. ", b: true }, { t: "보고서에 개별 환자 사례를 제시할 경우 식별 가능한 정보를 포함하지 않고, 필요 시 대표 궤적으로 대체한다." }]),
  BULLET([{ t: "외부 데이터의 이용 조건 준수. ", b: true }, { t: "MIMIC-IV는 제3자 공유를 금지하는 DUA 하에 제공되므로 원자료를 안심존에 반입하지 않고 파생 모델만 반입한다. PhysioNet CITI 인증은 취득 완료하였다." }]),
  H2("10.2 임상 적용에 대한 명시적 한계"),
  P("본 과제의 산출물은 연구 목적의 예측 모델이며 의료기기가 아니다. 다음을 분명히 한다."),
  BULLET("본 모델의 출력은 위험도 추정치이며, 진단이나 처치 지시가 아니다"),
  BULLET("경보는 의료진의 판단을 대체하지 않으며 보조 정보로만 제공된다"),
  BULLET("임상 도입을 위해서는 별도의 전향적 검증과 규제 절차가 필요하다"),
  BULLET("후향적 데이터로 산출된 성능은 실제 운영 환경의 성능을 보장하지 않는다"),
  H2("10.3 재현성 확보"),
  P("분석의 재현성은 신뢰성의 전제다. 분할 시드를 고정하고, 전처리·피처 생성·학습·평가 전 과정을 코드로 관리하며, 동일한 입력에 대해 동일한 결과가 재현되는지를 자동화된 테스트로 확인한다. 현재 파이프라인은 데이터 없이도 실행 가능한 합성 코호트를 내장하고 있어, 코드 자체의 동작 검증이 데이터 접근과 독립적으로 수행된다."),
  CALLOUT([
    { t: "폐쇄망은 재현성 요구를 강화한다. ", b: true },
    { t: "안심존 내부에서는 반복 실행 비용이 크고 외부 자원에 접근할 수 없으므로, 들어가기 전에 재현 가능한 상태로 코드를 완성해 두는 것이 실무적으로도 필수다." },
  ]),
  BREAK(),
);

// ── 10. 결론
children.push(
  H1("11. 결론"),
  P([
    { t: "본 제안은 정확도 경쟁이 아니라 " },
    { t: "오경보 저감, 설명가능성, 개인화", b: true },
    { t: "를 통해 임상 도입의 실제 장벽을 푸는, 가볍고 이식 가능한 원내 심정지 조기경보를 제안한다." },
  ]),
  P("활력징후로 심정지를 예측할 수 있다는 것은 이미 입증된 사실이다. 남아 있는 문제는 그 예측이 병동에서 신뢰받고 감당될 수 있는가이며, 본 제안은 그 질문에 대해 세 가지로 답한다 — 동일 검출률에서의 알람 부담을 1급 지표로 삼고, 모든 경보에 근거를 붙이며, 환자 개인의 기저선을 기준으로 판단한다."),
  P("경북대학교병원 활력징후 데이터의 case-only 구조는 within-patient 설계로 자연스럽게 활용되며, 그 구조가 만들어내는 측정의 한계까지 명시하고 보정 경로를 설계에 포함하였다. 파이프라인은 공개 실데이터에서 이미 end-to-end로 검증되었고 폐쇄망 실행 조건까지 확인되어 있어, 본선에서 즉시 실행 가능한 상태다."),
  CALLOUT([
    { t: "심정지는 갑작스러운 사건이 아니다. 전조를 남긴다. ", b: true },
    { t: "그 전조를 환자 자신의 기저선 기준으로 포착하고, 왜 위험한지와 함께 알리는 것 — 그것이 본 제안이 하려는 일이다." },
  ]),
  H2("11.1 본 제안서의 자체 점검"),
  P("심사 항목별로 본 제안서가 어디에서 답하고 있는지를 정리한다."),
  TABLE(
    [2000, 7026],
    ["심사 항목", "본 제안서의 대응"],
    [
      [[{ t: "시의성", b: true }], "제1장 — alarm fatigue로 인한 미충족 수요, 상용화 단계에 이른 시장에서 남아 있는 간극, 환자안전이라는 국민 건강 가치와의 연결"],
      [[{ t: "실현성", b: true }], "제5·6·7·9장 — 구체적 스키마 활용 설계, 실제 구현된 58개 피처 명세, 공개 실데이터 20,336명 end-to-end 검증, 배포된 화면, 확인 완료된 안심존 운영 조건과 그에 맞춘 작업 배치"],
      [[{ t: "참신성", b: true }], "제4·7장 — 개인 기저선 이탈, 경보 단위 근거 제시, case-only 구조의 within-patient 전환, 심정지 표현형 발견, 그리고 라벨 정의의 구조적 한계 발견과 이중 채널 대응"],
      [[{ t: "파급성", b: true }], "제8장 — 임상 도입 장벽 해소, 활력징후만으로 동작하는 낮은 도입 장벽, 타 병원 이식, 유사 구조 데이터셋으로의 방법론 재사용, 평가 프레임의 표준화 기여"],
    ],
  ),
  BREAK(),
);

// ── 참고문헌
children.push(
  H1("참고문헌"),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[1] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Royal College of Physicians. National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS. London: RCP, 2017.", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[2] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Kwon JM, Lee Y, Lee Y, et al. An Algorithm Based on Deep Learning for Predicting In-Hospital Cardiac Arrest. Journal of the American Heart Association. 2018;7(13):e008678. doi:10.1161/JAHA.118.008678", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[3] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Cho KJ, Kim JS, Lee DH, et al. Prospective, multicenter validation of the deep learning-based cardiac arrest risk management system for predicting in-hospital cardiac arrest or unplanned intensive care unit transfer in patients admitted to general wards. Critical Care. 2023;27(1):346. doi:10.1186/s13054-023-04609-0", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[4] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Agency for Healthcare Research and Quality. Alarm Fatigue. In: Making Healthcare Safer III: A Critical Analysis of Existing and Emerging Patient Safety Practices. Rockville, MD: AHRQ, 2020.", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[5] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE. 2015;10(3):e0118432. doi:10.1371/journal.pone.0118432", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[6] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS), 2017.", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[7] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[8] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2022.", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[9] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023;10:1.", size: 19, color: INK, font: FONT }) ] }),
  new Paragraph({ spacing: { after: 130, line: 280 }, indent: { left: 460, hanging: 460 },
    children: [ new TextRun({ text: "[10] ", bold: true, size: 19, color: ACCENT, font: FONT }),
                new TextRun({ text: "Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine. 2020;48(2):210-217.", size: 19, color: INK, font: FONT }) ] }),
  CALLOUT([
    { t: "〔제출 전 확인〕 ", b: true, c: WARN },
    { t: "상용 솔루션의 인허가 현황은 제품 표시·허가 정보가 갱신될 수 있으므로 제출 시점 기준으로 재확인하여 표기한다." },
  ], "FBF2F2", WARN),
);

// ─────────────────────────────────────────── document
const doc = new Document({
  creator: "PRODROME",
  title: "개인화 기저선 이탈 기반 설명가능 원내 심정지 조기경보",
  description: "2026 K-Health 미개방 의료데이터 활용 경진대회 예선 제안서",
  styles: {
    default: {
      document: { run: { font: FONT, size: 20, color: INK } },
      heading1: { run: { font: FONT } },
      heading2: { run: { font: FONT } },
      heading3: { run: { font: FONT } },
    },
  },
  sections: [{
    properties: {
      page: {
        margin: {
          top: convertInchesToTwip(1), bottom: convertInchesToTwip(1),
          left: convertInchesToTwip(1), right: convertInchesToTwip(1),
        },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: ["— ", PageNumber.CURRENT, " —"],
            size: 17, color: MUTED, font: FONT,
          })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, (buf.length / 1024).toFixed(0) + "KB");
});
