// End-of-chapter practice quizzes: instant feedback per question, no timer.
// A page opts in with <div class="practice-quiz" data-quiz="KEY"></div>,
// where window.PRACTICE_QUIZZES[KEY] is defined by a file in assets/js/quizzes/.
(function () {
  "use strict";

  var LETTERS = ["A", "B", "C", "D"];

  function init() {
    var roots = document.querySelectorAll(".practice-quiz[data-quiz]");
    roots.forEach(function (root) {
      var key = root.getAttribute("data-quiz");
      var quiz = (window.PRACTICE_QUIZZES || {})[key];
      if (!quiz) {
        root.innerHTML = "<p><em>Practice questions not loaded.</em></p>";
        return;
      }
      render(root, { quiz: quiz, answers: {} });
    });
  }

  function render(root, state) {
    root.innerHTML = "";
    root.appendChild(scoreBar(state));
    state.quiz.questions.forEach(function (q, i) {
      root.appendChild(questionCard(q, i + 1, state, root));
    });
    root.appendChild(resetControl(root, state));
  }

  function scoreBar(state) {
    var el = document.createElement("div");
    el.className = "pq-scorebar";
    var total = state.quiz.questions.length;
    var answered = 0;
    var correct = 0;
    state.quiz.questions.forEach(function (q) {
      if (state.answers[q.id]) {
        answered += 1;
        if (state.answers[q.id] === q.correct) correct += 1;
      }
    });
    var text = answered === 0
      ? total + " questions · pick an option to check your answer instantly"
      : correct + " correct out of " + answered + " answered · " + (total - answered) + " left";
    if (answered === total) {
      text = "Done: " + correct + " / " + total + " correct. " + verdict(correct / total);
    }
    el.textContent = text;
    return el;
  }

  function verdict(ratio) {
    if (ratio >= 0.8) return "Solid — move on to the next chapter.";
    if (ratio >= 0.6) return "Good — re-read the sections behind the ones you missed.";
    return "Re-read this chapter's exam-radar topics, then retry.";
  }

  function questionCard(q, num, state, root) {
    var picked = state.answers[q.id];
    var card = document.createElement("div");
    card.className = "pq-card" + (picked ? " is-answered" : "");

    var head = document.createElement("div");
    head.className = "pq-head";
    head.innerHTML =
      '<span class="pq-num">Q' + num + "</span>" +
      (q.type ? '<span class="pq-type">' + escape(q.type) + "</span>" : "");
    card.appendChild(head);

    var stem = document.createElement("div");
    stem.className = "pq-stem";
    stem.textContent = q.stem;
    card.appendChild(stem);

    if (q.code) {
      var pre = document.createElement("pre");
      pre.className = "pq-code";
      pre.innerHTML = "<code>" + escape(q.code) + "</code>";
      card.appendChild(pre);
    }

    var opts = document.createElement("div");
    opts.className = "pq-options";
    LETTERS.forEach(function (letter) {
      if (!q.options[letter]) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pq-option";
      btn.disabled = !!picked;
      if (picked) {
        if (letter === q.correct) btn.classList.add("is-correct");
        else if (letter === picked) btn.classList.add("is-wrong");
      }
      btn.innerHTML =
        '<span class="pq-letter">' + letter + "</span>" +
        '<span class="pq-option-text">' + escape(q.options[letter]) + "</span>";
      btn.addEventListener("click", function () {
        state.answers[q.id] = letter;
        render(root, state);
        var again = root.querySelectorAll(".pq-card")[num - 1];
        if (again) again.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
      opts.appendChild(btn);
    });
    card.appendChild(opts);

    if (picked) card.appendChild(feedback(q, picked));
    return card;
  }

  function feedback(q, picked) {
    var el = document.createElement("div");
    var right = picked === q.correct;
    el.className = "pq-feedback " + (right ? "correct" : "wrong");
    var html =
      '<div class="pq-status">' +
      (right ? "✓ Correct" : "✗ Not quite — the answer is " + q.correct) +
      "</div><ul class=\"pq-why\">";
    LETTERS.forEach(function (letter) {
      if (!q.why || !q.why[letter]) return;
      html +=
        '<li class="' + (letter === q.correct ? "why-correct" : "why-wrong") + '"><strong>' +
        letter + (letter === q.correct ? " ✓" : " ✗") + "</strong> " + escape(q.why[letter]) + "</li>";
    });
    html += "</ul>";
    if (q.takeaway) {
      html += '<div class="pq-takeaway"><strong>Remember:</strong> ' + escape(q.takeaway) + "</div>";
    }
    el.innerHTML = html;
    return el;
  }

  function resetControl(root, state) {
    var el = document.createElement("div");
    el.className = "pq-controls";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "md-button";
    btn.textContent = "Reset answers";
    btn.addEventListener("click", function () {
      state.answers = {};
      render(root, state);
      root.scrollIntoView({ behavior: "smooth" });
    });
    el.appendChild(btn);
    return el;
  }

  function escape(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
