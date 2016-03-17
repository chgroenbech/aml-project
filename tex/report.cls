\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{report}[2013/02/21 v0.8 Project class]

\usepackage{xifthen}					% Implements if clauses

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}

\ProcessOptions\relax

\LoadClass[a4paper,11pt,twoside,onecolumn,openany,article]{memoir}

% Load standard packages
\usepackage{fixltx2e}					% Fixes to LaTeX2e
\usepackage[utf8]{inputenc}				% Input encoding: UTF-8
\usepackage[T1]{fontenc}				% Font encoding: T1
\usepackage{graphicx}					% Graphics package

% Load localization packages
\usepackage{babel}						% For languages in document class
\usepackage{datetime}					% Include time commands
\usepackage[cleanlook]{isodate} 		% Date formatting: D MMMM YYYY

% Load typeface package
\usepackage[light, easyscsl, noDcommand]{kpfonts}
% Typeface: The Kepler project
% Also loads ASM packages

% Load math packages
\usepackage{mathtools}					% Extra math commands
\usepackage{bm}							% Bold math
\usepackage{slashed}

% Load packages for scientific articles
\usepackage{siunitx}					% SI units
\usepackage[round,semicolon]{natbib}	% Natural science refereneces

% Load programming packages
\usepackage{listings}					% Include source code
\usepackage{algorithmic}				% Typeset algorithms
\usepackage[chapter]{algorithm}			% Float for algorithms

% Load miscellaneous packgaes
\usepackage{xcolor}						% Extended package for colouring
\usepackage{ellipsis}					% Fixes ellipses before letters
\usepackage{xspace}						% Add space after text macros with \xspace
\usepackage{paralist}					% Lists in paragraphs
\usepackage{afterpage}
\usepackage{flafter}
\usepackage{grffile}
\usepackage{keyval}

% Load cross reference packages
\usepackage{hyperref}					% PDF links for cross-references, citations, etc.
\usepackage{bookmark}					% Fixes aspects of the hyperref package

% Note: For the hyperref package to work properly it should be loaded after all other packages except the bookmark package, but before \newcommand's and such.

% Colours
\definecolor{dark-red}{rgb}{0.4,0.15,0.15}
\definecolor{dark-blue}{rgb}{0.15,0.15,0.4}
\definecolor{medium-blue}{rgb}{0,0,0.5}
\definecolor{light-grey}{rgb}{0.98,0.98,0.98}

% Page setup
\settrimmedsize{297mm}{210mm}{*}
\setlength{\trimtop}{0pt}
\setlength{\trimedge}{\stockwidth}
\addtolength{\trimedge}{-\paperwidth}
% \settypeblocksize{*}{400pt}{1.618}
% \setulmargins{4cm}{*}{*}
\settypeblocksize{660pt}{450pt}{*}
\setulmargins{3.5cm}{*}{*}
% \setlrmargins{*}{*}{1}
\setlrmargins{*}{*}{1.5}
\setmarginnotes{0pt}{0pt}{\onelineskip}
\setheadfoot{\onelineskip}{2\onelineskip}
\setheaderspaces{*}{2\onelineskip}{*}
\checkandfixthelayout

% Title
\setlength{\droptitle}{-5ex}
\newcommand{\course}[1]{\gdef\thecourse{#1}}
\newcommand{\thecourse}{}
\renewcommand{\maketitlehooka}{%
	\par\noindent {\sffamily \scshape \large
		\thecourse
	} \vspace*{-3ex}
}
\pretitle{\begin{flushleft} \LARGE \bfseries \sffamily}
\posttitle{\end{flushleft} \vspace*{0ex}}
\preauthor{\begin{flushleft}
			\large \itshape \sffamily
			\begin{tabular}[t]{@{}l}}
\postauthor{\end{tabular}\end{flushleft}\vspace*{-1ex}}
\renewcommand*\and{%
	\end{tabular}%
	\hskip 1em \relax
	\begin{tabular}[t]{l}}
\renewcommand*{\andnext}{%
	\end{tabular}\\ \vspace*{-0.5ex} \begin{tabular}[t]{@{}l}}
\predate{\begin{flushleft} \normalsize \sffamily}
\postdate{\end{flushleft}}
\renewcommand{\maketitlehookd}{%
	\aftergroup\@afterindentfalse
	\aftergroup\@afterheading
}

% Page style
\makepagestyle{nice}
\makeevenhead {nice}{\sffamily \theauthor}{}{}
\makeoddhead  {nice}{}{}{\sffamily \thecourse}
\makeheadrule {nice}{\textwidth}{\normalrulethickness}
\makeevenfoot {nice}{\sffamily\thepage}{}{}
\makeoddfoot  {nice}{}{}{\sffamily\thepage}
\makepsmarks  {nice}{%
	\clearmark{chapter}
	\clearmark{section}
	\clearmark{subsection}
	\clearmark{subsubsection}
	\clearplainmark{toc}
	\clearplainmark{lot}
	\clearplainmark{lof}
	\clearplainmark{bib}
	\clearplainmark{index}
}
\nouppercaseheads % No upper cases.

\copypagestyle{plain}{nice}
\makeevenhead {plain}{}{}{}
\makeoddhead  {plain}{}{}{}
\makeheadrule {plain}{\textwidth}{0pt}
\makeevenfoot {plain}{\sffamily\thepage}{}{}
\makeoddfoot  {plain}{}{}{\sffamily\thepage}

\mergepagefloatstyle{nicefloats}{nice}{plain}

\pagestyle{nicefloats}

% Document division
\setsecnumdepth{subparagraph}
\maxsecnumdepth{subparagraph}

\renewcommand*{\chaptitlefont}{\normalfont\LARGE\bfseries}

% ToC
\settocdepth{subparagraph}
\maxtocdepth{subparagraph}

% Floats and other input
\captionnamefont{\bfseries}
\captiondelim{\ \ }
\captiontitlefinal{}
\newsubfloat{figure}
\tightsubcaptions

% Math
\allowdisplaybreaks
\newcommand{\idx}[1]{_{\text{#1}}}
\newcommand{\udx}[1]{^{\text{#1}}}
\newcommand{\order}[1]{^{(#1)}}
\let\dor\order
\DeclareMathOperator{\pr}{Pr}
\DeclareMathOperator{\E}{E}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator{\argmax}{arg \, max}
\DeclareMathOperator{\erf}{erf}
\DeclareMathOperator{\cond}{cond}
\DeclareMathOperator{\arctanh}{arctanh}
\DeclareMathOperator{\tracesymbol}{Tr}
\DeclareMathOperator{\AR}{AR}
\DeclareMathOperator{\MA}{MA}
\DeclareMathOperator{\ARMA}{ARMA}
\DeclareMathOperator{\ARIMA}{ARIMA}
% \DeclareMathSymbol{\diff}{\mathord}{letters}{`\d}
\DeclareMathSymbol{\diff}{\mathord}{operators}{`\d}
\DeclareMathSymbol{\backshift}{\mathord}{operators}{`\B}
\DeclareMathSymbol{\summation}{\mathord}{operators}{`\S}
\DeclareMathSymbol{\discretediff}{\mathord}{operators}{`\D}
\DeclareMathSymbol{\eup}{\mathord}{operators}{`\e}
\DeclareMathSymbol{\iup}{\mathord}{operators}{`\i}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\difference}[1]{\Delta #1}
\newcommand{\D}[2][]{
	\ifthenelse{\isempty{#1}}{\mathclose{\diff \! #2}}
	{\mathclose{\diff^{#1} \! #2}}
}
\newcommand{\DD}[2][]{
	\ifthenelse{\isempty{#1}}{\discretediff \! #2}
	{\discretediff_{#1} \! #2}
}
\newcommand{\deriv}[3][]{
	\ifthenelse{\isempty{#1}}{\frac{\diff \! #2}{\diff \! #3}}
	{\frac{\diff^{#1} \! #2}{\diff \! #3^{#1}}}
}
\newcommand{\pderiv}[3][]{
	\ifthenelse{\isempty{#1}}{\frac{\partial \! #2}{\partial \! #3}}
	{\frac{\partial^{#1} \! #2}{\partial \! #3^{#1}}}
}
\newcommand{\mpderiv}[3]{
	\frac{\partial^2 \! #1}{\partial \! #2 \partial \! #3}
}
\newcommand{\grad}[2][]{
	\ifthenelse{\isempty{#1}}{\nabla #2}{\nabla_{#1} #2}
}
\newcommand{\divg}[1]{\nabla \cdot #1}
\newcommand{\curl}[1]{\nabla \times #1}
\newcommand{\Hessian}[1]{\nabla^2 #1}
\let\oldvec\vec
\renewcommand{\vec}[1]{\bm{\mathrm{#1}}}
\newcommand{\psvec}[1]{\tilde{\vec{#1}}}
\newcommand{\uvec}[1]{\hat{\bm{\mathrm{#1}}}}
\newcommand{\gvec}[1]{\oldvec{#1}}
\newcommand{\mat}[1]{\bm{\mathrm{#1}}}
\newcommand{\transpose}[2][]{
	\ifthenelse{\isempty{#1}}{{#2}^\mathsf{T}}{{#2}^{#1\mathsf{T}}}
}
\let\T\transpose
\newcommand{\Hermitian}[1]{{#1}^\dagger\!}
\newcommand{\h}[1]{\Hermitian{#1}}
\newcommand{\conjugate}[1]{\overline{#1}}
\let\oldhat\hat
\renewcommand{\hat}[1]{\oldhat{#1\!\,}}
\let\olddot\dot
\renewcommand{\dot}[1]{\olddot{#1\!\,}}
\newcommand{\op}[1]{\hat{#1}}
\newcommand{\modf}[1]{\widetilde{#1}}
\newcommand{\convolution}[2]{#1 \ast #2}
\newcommand{\correlation}[2]{#1 \star #2}
\DeclarePairedDelimiter{\expval}{\langle}{\rangle}
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\detm}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\DeclarePairedDelimiter{\parens}{(}{)}
\DeclarePairedDelimiter{\brackets}{[}{]}
\DeclarePairedDelimiter{\curlies}{\{}{\}}
\DeclarePairedDelimiter{\set}{\{}{\}}
\DeclarePairedDelimiter{\bra}{\langle}{\rvert}
\DeclarePairedDelimiter{\ket}{\lvert}{\rangle}
\DeclarePairedDelimiter{\ceiling}{\lceil}{\rceil}
\DeclarePairedDelimiter{\floor}{\lfloor}{\rfloor}
\DeclarePairedDelimiterX{\braket}[2]{\langle}{\rangle}{#1 \vert #2}
\DeclarePairedDelimiterX{\commutator}[2]{[}{]}{#1, #2}
\DeclarePairedDelimiterX{\anticommutator}[2]{\{}{\}}{#1, #2}
\DeclarePairedDelimiterX{\interval}[2]{[}{]}{#1, #2}
\newcommand{\trace}{\@ifstar\@tracestar\@trace}
\newcommand{\@trace}[2][]{\tracesymbol \brackets[#1]{#2}}
\newcommand{\@tracestar}[1]{\tracesymbol \brackets*{#1}}
% \newcommand{\s}[1]{#1\mathllap{/}}
\newcommand{\s}[1]{\slashed{#1}}
\newcommand{\mean}[1]{\bar{#1}}
\newcommand{\var}[1]{\sigma_{#1}^2}
\newcommand{\width}[1]{\sigma_{#1}}
\newcommand{\given}{\, | \,}
\newcommand{\constant}{\text{const.}}
\newcommand{\logicand}{\wedge}
\newcommand{\logicor}{\vee}


% Units
\sisetup{%
	per=slash,
	round-precision = 5,
	round-mode = off,
	range-units = single,
	range-phrase = \text{--},
	list-separator = {\text{, }},
	list-final-separator = {\text{, and }}
}
\DeclareSIUnit{\deg}{deg}
\DeclareSIUnit{\year}{yr}

% Code
\lstset{
language=Python,                % language of code
basicstyle=\footnotesize\ttfamily,       % size of fonts used for code
% columns=flexible,
numbers=left,                   % where to put line numbers
numberstyle=\footnotesize,      % size of fonts used for line numbers
stepnumber=1,                   % step between two line numbers
numbersep=5pt,                  % how far line numbers are from code
backgroundcolor=\color{light-grey},  % background color
showspaces=false,               % underline spaces
showstringspaces=false,         % underline spaces within strings
showtabs=false,                 % underline tabs within strings
frame=single,                   % adds a frame around the code
tabsize=2,                      % sets default tabsize to 2 spaces
captionpos=b,                   % sets the caption-position to bottom
breaklines=true,                % automatic line breaking
breakatwhitespace=false,        % automatic breaks only at whitespace
% title=		                 % filename of files included with \lstinputlisting
% escapeinside={\%*}{*)},         % if you want to add a comment within your code
morekeywords={as}            % if you want to add more keywords to the set
}

% Revisioning
\newcommand{\change}[1]{\textcolor{red}{\emph{(#1)}}}
\newcommand{\rephrase}[1]{\textcolor{orange}{#1}}

% References
\addto\captionsbritish{\renewcommand\bibname{References}}

% Text formatting
\newcommand{\newterm}[1]{\textbf{#1}}
\newcommand{\programminglanguage}[1]{\textsf{#1}}
\newcommand{\library}[1]{\textsf{#1}}
\newcommand{\function}[1]{\texttt{#1}}
\newcommand{\class}[1]{\textbf{\texttt{#1}}}
\newcommand{\file}[1]{\texttt{#1}}
\newcommand{\variable}[1]{\textit{\texttt{#1}}}
\newcommand{\code}[1]{\texttt{#1}}

% Nomenclature
\newcommand{\Q}{\protect\othertailQ uestion}
\newcommand{\Qs}{\Q s}

% Cross references
\hypersetup{
	colorlinks,
	linkcolor={dark-red},
	citecolor={dark-blue},
	urlcolor={medium-blue},
	% bookmarks=true,
	bookmarksnumbered=true,
	unicode=true,
	plainpages=false,
	% pdfborder={0 0 0},
}
\bookmarksetup{startatroot}
\AtBeginDocument{%
	\hypersetup{pdftitle={\thetitle}, pdfauthor={\theauthor}, pdfsubject={\thecourse}}
}
