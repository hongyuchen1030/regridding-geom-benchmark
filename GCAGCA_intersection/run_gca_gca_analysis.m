(* gca_gca_errors.m — baseline + relative error; write CSV w/o quotes; print first 5 *)

(* ---------- Robust CLI (last 3 args) ---------- *)
getArgs[] := Module[{sc = If[ValueQ[$ScriptCommandLine] && ListQ[$ScriptCommandLine], $ScriptCommandLine, {}], args = {}},
  If[MemberQ[sc, "-script"],
    With[{p = First@FirstPosition[sc, "-script", Missing["NF"]]},
      If[p =!= Missing["NF"], sc = Drop[sc, p + 1]]
    ]
  ];
  If[Length[sc] >= 1 && StringMatchQ[ToString@sc[[1]], ___ ~~ ".m"], sc = Rest[sc]];
  args = sc;
  If[Length[args] < 3 && ListQ[$CommandLine],
    args = $CommandLine;
    If[MemberQ[args, "-script"],
      With[{p = First@FirstPosition[args, "-script", Missing["NF"]]},
        If[p =!= Missing["NF"], args = Drop[args, p + 1]]
      ]
    ];
    If[Length[args] >= 1 && StringMatchQ[ToString@args[[1]], ___ ~~ ".m"], args = Rest[args]];
  ];
  If[Length[args] >= 3, args = args[[-3 ;;]]];
  args
];

args = getArgs[];
If[Length[args] != 3,
  Print["Usage: math -script gca_gca_errors.m <arcsPath> <resPath> <outPath>"];
  Print["Received args: ", args];
  Quit[1];
];
{arcsPath, resPath, outPath} = args;

Print["[INFO] arcsPath = ", arcsPath];
Print["[INFO] resPath  = ", resPath];
Print["[INFO] outPath  = ", outPath];

(* ---------- Helpers ---------- *)
wp = 50;

safeData[path_] := Module[{raw},
  raw = Quiet@Check[Import[path, {"CSV", "Data"}], $Failed];
  If[raw === $Failed || !ListQ[raw] || Length[raw] == 0, Return[{}]];
  If[ListQ[First[raw]], Rest[raw], raw]
];

asInt[x_] := Module[{y = x},
  Which[
    IntegerQ[y], y,
    StringQ[y], Quiet@Check[ToExpression[StringTrim[y]], 0],
    True, IntegerPart @ Quiet@Check[N[y], 0]
  ]
];

asReal[x_] := Module[{y = x},
  Which[
    NumberQ[y], N[y, wp],
    StringQ[y], Quiet@Check[N@ToExpression[StringTrim[y], InputForm, HoldForm] /. HoldForm[z_] :> z, 0., {ToExpression::sntxi}],
    True, N[0., wp]
  ]
];

fromSigExp2[sig_, exp_] := SetPrecision[asInt[sig]*2^asInt[exp], wp];

vec3FromRow[row_List, start_Integer] := Module[
  {n = Length[row]},
  If[n < start + 5,
    {SetPrecision[0, wp], SetPrecision[0, wp], SetPrecision[0, wp]},
    {
      fromSigExp2[row[[start]],     row[[start + 1]]],
      fromSigExp2[row[[start + 2]], row[[start + 3]]],
      fromSigExp2[row[[start + 4]], row[[start + 5]]]
    }
  ]
];

normalizeWP[v_List] := Module[{n = Norm[v]},
  If[n == 0 || Not[NumericQ[n]],
    ConstantArray[SetPrecision[0, wp], 3],
    v/n
  ]
];

relErr[v_, vref_] := Module[{e1 = Norm[v - vref], e2 = Norm[v + vref]}, Min[e1, e2]];

toMantExp10[err_] := Module[{e = N[err, wp], exp10, mant},
  If[e == 0, {0, 0},
    exp10 = Floor[Log10[e]];
    mant  = e/10.^exp10;
    {mant, exp10}
  ]
];

numStr[x_] := ToString@NumberForm[x, 17, NumberPadding -> {"",""}, ExponentFunction -> (Null&)];

(* ---------- Read CSVs ---------- *)
arcsRows = safeData[arcsPath];
If[arcsRows === {}, Print["[ERROR] Failed to read ARCS CSV or it is empty: ", arcsPath]; Quit[2]];
validArcs = Select[arcsRows, Length[#] >= 26 &];
Print["[INFO] ARCS rows: ", Length[arcsRows], " (kept ≥26 cols: ", Length[validArcs], ")"];

resRows = safeData[resPath];
If[resRows === {}, Print["[ERROR] Failed to read RESULTS CSV or it is empty: ", resPath]; Quit[3]];
validRes = Select[resRows, Length[#] >= 20 &];
Print["[INFO] RESULTS rows: ", Length[resRows], " (kept ≥20 cols: ", Length[validRes], ")"];

(* ---------- Build baseline (Association pid -> vref) ---------- *)
baselineAssoc = Association[];
Do[
  Module[{row = validArcs[[i]], pidStr, pid, a0, a1, b0, b1, vref},
    pidStr = ToString[row[[1]]];
    pid    = asInt[row[[1]]]; (* numeric id for output *)
    a0 = vec3FromRow[row, 3];
    a1 = vec3FromRow[row, 9];
    b0 = vec3FromRow[row, 15];
    b1 = vec3FromRow[row, 21];
    vref = normalizeWP[ Cross[ Cross[a0, a1], Cross[b0, b1] ] ];
    baselineAssoc[pidStr] = vref;
  ],
  {i, 1, Length[validArcs]}
];

(* ---------- Compute errors (direct/kahan/eft) ---------- *)
headerLine = StringRiffle[
  {
    "pairs_id","ref_angle",
    "direct_error_mant10","direct_error_exp10",
    "kahan_error_mant10","kahan_error_exp10",
    "eft_error_mant10","eft_error_exp10"
  }, ","
];

outRows = {};
Do[
  Module[{row = validRes[[i]], pidStr, pidNum, refdeg, vref, vdir, vkahan, veft,
          derr, kerr, eerr, dm, de, km, ke, em, ee, outRow},
    pidStr = ToString[row[[1]]];
    pidNum = asInt[row[[1]]];
    refdeg = N[asReal[row[[2]]], 17];

    If[KeyExistsQ[baselineAssoc, pidStr],
      vref   = baselineAssoc[pidStr];
      vdir   = vec3FromRow[row, 3];
      vkahan = vec3FromRow[row, 9];
      veft   = vec3FromRow[row, 15];

      derr = relErr[vdir,   vref];
      kerr = relErr[vkahan, vref];
      eerr = relErr[veft,   vref];

      {dm, de} = toMantExp10[derr];
      {km, ke} = toMantExp10[kerr];
      {em, ee} = toMantExp10[eerr];

      outRow = {
        ToString[pidNum],             (* numeric id -> no quotes *)
        numStr[refdeg],
        numStr[N[dm, 17]], ToString[de],
        numStr[N[km, 17]], ToString[ke],
        numStr[N[em, 17]], ToString[ee]
      };
      AppendTo[outRows, outRow];
    ];
  ],
  {i, 1, Length[validRes]}
];

(* ---------- Print first 5 output rows ---------- *)
nshow = Min[5, Length[outRows]];
If[nshow == 0,
  Print["[WARN] No error rows to display."],
  Print["[INFO] First ", nshow, " error rows (", headerLine, "):"];
  Do[Print[outRows[[i]]], {i, 1, nshow}]
];

(* ---------- Write CSV manually (no quotes anywhere) ---------- *)
strm = OpenWrite[outPath, CharacterEncoding -> "UTF-8"];
WriteString[strm, headerLine <> "\n"];
Do[
  WriteString[strm, StringRiffle[outRows[[i]], ","] <> "\n"],
  {i, 1, Length[outRows]}
];
Close[strm];

Print["Wrote errors CSV (no quotes): ", outPath];
Quit[0];
