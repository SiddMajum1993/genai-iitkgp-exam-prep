(function () {
  "use strict";

  function init() {
    var root = document.getElementById("quiz-root");
    if (!root) return;
    var examName = root.getAttribute("data-exam");
    var exam = window[examName];
    if (!exam) {
      root.innerHTML = "<p><em>Quiz data not loaded.</em></p>";
      return;
    }

    var state = {
      exam: exam,
      answers: {},
      submitted: false,
      startedAt: null,
      endedAt: null,
      timerId: null,
      remainingSeconds: exam.durationMinutes * 60
    };

    render(root, state);
  }

  function render(root, state) {
    root.innerHTML = "";
    root.appendChild(header(state));
    root.appendChild(controls(root, state));

    var questionCounter = 0;
    state.exam.sections.forEach(function (section, sIdx) {
      root.appendChild(sectionHeader(section, sIdx));
      section.questions.forEach(function (q) {
        questionCounter += 1;
        root.appendChild(questionCard(q, questionCounter, state, root));
      });
    });

    root.appendChild(controls(root, state));
    root.appendChild(scorePanel(state));
  }

  function header(state) {
    var el = document.createElement("div");
    el.className = "quiz-header";
    var totalQs = state.exam.sections.reduce(function (n, s) {
      return n + s.questions.length;
    }, 0);
    el.innerHTML =
      '<h2 style="margin-top:0">' + escape(state.exam.title) + "</h2>" +
      '<p><strong>Format:</strong> ' + totalQs + " MCQs · " +
      state.exam.marksPerQuestion + " marks each · No negative marking · " +
      state.exam.durationMinutes + " minutes</p>" +
      '<div class="quiz-timer" id="quiz-timer">Timer not started</div>';
    return el;
  }

  function sectionHeader(section, sIdx) {
    var el = document.createElement("h3");
    el.className = "quiz-section-header";
    el.textContent = section.name;
    return el;
  }

  function questionCard(q, num, state, root) {
    var card = document.createElement("div");
    card.className = "quiz-card";
    card.id = "card-" + q.id;

    var stem = document.createElement("div");
    stem.className = "quiz-stem";
    stem.innerHTML = "<strong>Q" + num + ".</strong> " + escape(q.stem);
    card.appendChild(stem);

    if (q.code) {
      var pre = document.createElement("pre");
      pre.className = "quiz-code";
      pre.innerHTML = "<code>" + escape(q.code) + "</code>";
      card.appendChild(pre);
    }

    var opts = document.createElement("div");
    opts.className = "quiz-options";
    ["A", "B", "C", "D"].forEach(function (letter) {
      var text = q.options[letter];
      if (!text) return;
      var label = document.createElement("label");
      label.className = "quiz-option";
      label.setAttribute("data-letter", letter);
      var input = document.createElement("input");
      input.type = "radio";
      input.name = q.id;
      input.value = letter;
      if (state.answers[q.id] === letter) input.checked = true;
      input.disabled = state.submitted;
      input.addEventListener("change", function () {
        state.answers[q.id] = letter;
        if (!state.startedAt && !state.submitted) startTimer(state);
        updateProgress(state);
      });
      var letterSpan = document.createElement("span");
      letterSpan.className = "quiz-letter";
      letterSpan.textContent = letter;
      var textSpan = document.createElement("span");
      textSpan.className = "quiz-option-text";
      textSpan.textContent = text;
      label.appendChild(input);
      label.appendChild(letterSpan);
      label.appendChild(textSpan);
      opts.appendChild(label);
    });
    card.appendChild(opts);

    if (state.submitted) {
      card.appendChild(feedback(q, state));
      applyGrading(card, q, state);
    }

    return card;
  }

  function feedback(q, state) {
    var picked = state.answers[q.id];
    var correct = q.correct;
    var isRight = picked === correct;
    var el = document.createElement("div");
    el.className = "quiz-feedback " + (isRight ? "correct" : (picked ? "wrong" : "skipped"));
    var status = isRight
      ? "✓ Correct"
      : picked
        ? "✗ You picked " + picked + " — correct answer is " + correct
        : "— Skipped — correct answer is " + correct;
    el.innerHTML =
      '<div class="quiz-status">' + status + "</div>" +
      '<div class="quiz-explanation"><strong>Explanation:</strong> ' + escape(q.explanation) + "</div>" +
      (q.reference ? '<div class="quiz-reference"><em>' + escape(q.reference) + "</em></div>" : "");
    return el;
  }

  function applyGrading(card, q, state) {
    var picked = state.answers[q.id];
    var labels = card.querySelectorAll(".quiz-option");
    labels.forEach(function (l) {
      var letter = l.getAttribute("data-letter");
      if (letter === q.correct) l.classList.add("is-correct");
      if (letter === picked && picked !== q.correct) l.classList.add("is-wrong");
    });
  }

  function controls(root, state) {
    var el = document.createElement("div");
    el.className = "quiz-controls";
    if (!state.submitted) {
      var submitBtn = document.createElement("button");
      submitBtn.className = "md-button md-button--primary";
      submitBtn.type = "button";
      submitBtn.textContent = "Submit exam & see score";
      submitBtn.addEventListener("click", function () {
        var unanswered = countUnanswered(state);
        if (unanswered > 0) {
          if (!confirm(unanswered + " question(s) are unanswered. Submit anyway?")) return;
        }
        submit(state);
        render(root, state);
        window.scrollTo({ top: root.offsetTop, behavior: "smooth" });
      });
      el.appendChild(submitBtn);

      var progress = document.createElement("span");
      progress.className = "quiz-progress";
      progress.id = "quiz-progress";
      progress.textContent = progressText(state);
      el.appendChild(progress);
    } else {
      var resetBtn = document.createElement("button");
      resetBtn.className = "md-button";
      resetBtn.type = "button";
      resetBtn.textContent = "Reset & retake";
      resetBtn.addEventListener("click", function () {
        if (!confirm("Reset all answers and retake the exam?")) return;
        state.answers = {};
        state.submitted = false;
        state.startedAt = null;
        state.endedAt = null;
        state.remainingSeconds = state.exam.durationMinutes * 60;
        if (state.timerId) { clearInterval(state.timerId); state.timerId = null; }
        render(root, state);
        window.scrollTo({ top: root.offsetTop, behavior: "smooth" });
      });
      el.appendChild(resetBtn);
    }
    return el;
  }

  function scorePanel(state) {
    var el = document.createElement("div");
    el.className = "quiz-score-panel";
    if (!state.submitted) { el.style.display = "none"; return el; }

    var totalQs = 0;
    var correctCount = 0;
    var perSection = [];
    state.exam.sections.forEach(function (section) {
      var secTotal = section.questions.length;
      var secCorrect = 0;
      section.questions.forEach(function (q) {
        totalQs += 1;
        if (state.answers[q.id] === q.correct) {
          correctCount += 1;
          secCorrect += 1;
        }
      });
      perSection.push({ name: section.name, total: secTotal, correct: secCorrect });
    });

    var marks = correctCount * state.exam.marksPerQuestion;
    var maxMarks = totalQs * state.exam.marksPerQuestion;
    var pct = Math.round((correctCount / totalQs) * 100);

    var band;
    if (pct >= 83) band = { text: "Excellent — you are ready", cls: "band-excellent" };
    else if (pct >= 70) band = { text: "Strong — confident for real exam", cls: "band-strong" };
    else if (pct >= 50) band = { text: "Pass — but review weak areas", cls: "band-pass" };
    else band = { text: "Below passing — more study needed", cls: "band-fail" };

    var duration = state.endedAt && state.startedAt
      ? formatDuration(Math.round((state.endedAt - state.startedAt) / 1000))
      : "—";

    var rows = perSection.map(function (s) {
      var scPct = Math.round((s.correct / s.total) * 100);
      return '<tr><td>' + escape(s.name) + '</td><td style="text-align:right">' +
        s.correct + " / " + s.total + '</td><td style="text-align:right">' + scPct + "%</td></tr>";
    }).join("");

    el.innerHTML =
      '<h3 style="margin-top:0">Your score</h3>' +
      '<div class="quiz-score-big"><span class="quiz-score-num">' + correctCount + " / " + totalQs + '</span>' +
      '<span class="quiz-score-marks">' + marks + " / " + maxMarks + ' marks · ' + pct + '%</span></div>' +
      '<div class="quiz-band ' + band.cls + '">' + band.text + '</div>' +
      '<p><strong>Time taken:</strong> ' + duration + "</p>" +
      '<h4>Per-section breakdown</h4>' +
      '<table class="quiz-section-table"><thead><tr><th>Section</th><th style="text-align:right">Score</th><th style="text-align:right">%</th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table>' +
      '<p style="margin-top:1em"><strong>Grading guide:</strong></p>' +
      '<ul>' +
      '<li>12/24 (48 marks) = Pass (borderline)</li>' +
      '<li>17/24 (68 marks) = Healthy pass (recommended target)</li>' +
      '<li>20/24 (80 marks) = Strong performance</li>' +
      '</ul>' +
      '<p><em>Review each wrong answer below — the correct choice is highlighted green, your incorrect pick is highlighted red, and the explanation is expanded.</em></p>';
    return el;
  }

  function countUnanswered(state) {
    var n = 0;
    state.exam.sections.forEach(function (section) {
      section.questions.forEach(function (q) {
        if (!state.answers[q.id]) n += 1;
      });
    });
    return n;
  }

  function progressText(state) {
    var totalQs = 0;
    var answered = 0;
    state.exam.sections.forEach(function (section) {
      section.questions.forEach(function (q) {
        totalQs += 1;
        if (state.answers[q.id]) answered += 1;
      });
    });
    return answered + " / " + totalQs + " answered";
  }

  function updateProgress(state) {
    var el = document.getElementById("quiz-progress");
    if (el) el.textContent = progressText(state);
  }

  function submit(state) {
    state.submitted = true;
    state.endedAt = Date.now();
    if (state.timerId) { clearInterval(state.timerId); state.timerId = null; }
  }

  function startTimer(state) {
    state.startedAt = Date.now();
    updateTimerDisplay(state);
    state.timerId = setInterval(function () {
      var elapsed = Math.floor((Date.now() - state.startedAt) / 1000);
      state.remainingSeconds = state.exam.durationMinutes * 60 - elapsed;
      updateTimerDisplay(state);
      if (state.remainingSeconds <= 0) {
        clearInterval(state.timerId);
        state.timerId = null;
        alert("Time's up! Submitting your exam.");
        submit(state);
        var root = document.getElementById("quiz-root");
        render(root, state);
      }
    }, 1000);
  }

  function updTimer() { /* legacy */ }

  function updateTimerDisplay(state) {
    var el = document.getElementById("quiz-timer");
    if (!el) return;
    var s = state.remainingSeconds;
    if (s < 0) s = 0;
    var mins = Math.floor(s / 60);
    var secs = s % 60;
    var label = "Time remaining: " + pad(mins) + ":" + pad(secs);
    el.textContent = label;
    el.classList.toggle("timer-warn", s <= 600 && s > 60);
    el.classList.toggle("timer-danger", s <= 60);
  }

  function pad(n) { return n < 10 ? "0" + n : "" + n; }

  function formatDuration(sec) {
    var m = Math.floor(sec / 60);
    var s = sec % 60;
    return m + " min " + s + " sec";
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
