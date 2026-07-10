window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(function () {
  if (window.MathJax && MathJax.typesetPromise) {
    try {
      if (MathJax.startup && MathJax.startup.output && MathJax.startup.output.clearCache) {
        MathJax.startup.output.clearCache();
      }
      if (MathJax.typesetClear) MathJax.typesetClear();
      if (MathJax.texReset) MathJax.texReset();
      MathJax.typesetPromise();
    } catch (e) {
      // MathJax not fully ready or no math on page — safe to skip
    }
  }
});
