// Force all LinkedIn links (header identity chip, About page button, etc.)
// to open in a new browser tab so the study guide stays put.
(function () {
  function rewireLinkedIn() {
    document
      .querySelectorAll('a[href*="linkedin.com"], a.md-source')
      .forEach(function (a) {
        a.setAttribute("target", "_blank");
        a.setAttribute("rel", "noopener noreferrer");
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", rewireLinkedIn);
  } else {
    rewireLinkedIn();
  }

  // Material's instant-loading swaps the DOM without a full reload.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(rewireLinkedIn);
  }

  // Belt-and-braces: delegate clicks in case a link was added after our patch.
  document.addEventListener(
    "click",
    function (e) {
      var a = e.target.closest && e.target.closest("a");
      if (!a) return;
      var href = a.getAttribute("href") || "";
      if (href.indexOf("linkedin.com") === -1) return;
      if (a.getAttribute("target") === "_blank") return; // already handled
      e.preventDefault();
      window.open(href, "_blank", "noopener,noreferrer");
    },
    true
  );
})();
