document.addEventListener("DOMContentLoaded", function () {

  // ---------------- Symptom chip search + running counter ----------------
  const searchInput = document.getElementById("symptom-search");
  const symptomChips = document.querySelectorAll(".symptom-chip");
  const counter = document.getElementById("symptom-counter");
  const checkboxes = document.querySelectorAll(".symptom-checkbox");

  function updateCounter() {
    if (!counter) return;
    const checkedCount = document.querySelectorAll(".symptom-checkbox:checked").length;
    counter.textContent = checkedCount + " selected";
  }

  checkboxes.forEach((cb) => cb.addEventListener("change", updateCounter));
  updateCounter();

  function filterSymptoms(query) {
    const q = query.trim().toLowerCase();
    let visibleCount = 0;
    symptomChips.forEach((chip) => {
      const name = chip.getAttribute("data-name") || "";
      const isMatch = name.includes(q);
      chip.style.display = isMatch ? "inline-flex" : "none";
      if (isMatch) visibleCount++;
    });
    const noMatchMsg = document.getElementById("no-symptom-match");
    if (noMatchMsg) {
      noMatchMsg.style.display = (q.length > 0 && visibleCount === 0) ? "block" : "none";
    }
  }

  if (searchInput) {
    searchInput.addEventListener("input", function () {
      filterSymptoms(searchInput.value);
    });
  }

  // ---------------- Voice-assisted symptom search ----------------
  // Uses the browser's built-in Web Speech API. Silently does nothing on
  // browsers that don't support it (feature-detected, no crash).
  const voiceBtn = document.getElementById("voice-search-btn");
  const SpeechRecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (voiceBtn) {
    if (!SpeechRecognitionClass) {
      voiceBtn.style.display = "none";
    } else {
      const recognition = new SpeechRecognitionClass();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternates = 1;

      voiceBtn.addEventListener("click", function () {
        voiceBtn.classList.add("listening");
        recognition.start();
      });

      recognition.addEventListener("result", function (event) {
        const transcript = event.results[0][0].transcript;
        if (searchInput) {
          searchInput.value = transcript;
          filterSymptoms(transcript);
        }
      });

      recognition.addEventListener("end", function () {
        voiceBtn.classList.remove("listening");
      });

      recognition.addEventListener("error", function () {
        voiceBtn.classList.remove("listening");
      });
    }
  }

  // ---------------- Quick temperature buttons (no thermometer needed) ----------------
  const quickTempBtns = document.querySelectorAll(".quick-temp-btn");
  const temperatureInput = document.getElementById("temperature-input");

  quickTempBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      quickTempBtns.forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");

      // Clear any previously auto-checked fever symptom
      ["mild_fever", "high_fever"].forEach((key) => {
        const cb = document.querySelector(`.symptom-checkbox[value="${key}"]`);
        if (cb) cb.checked = false;
      });

      const symptomKey = btn.getAttribute("data-symptom");
      if (symptomKey) {
        const cb = document.querySelector(`.symptom-checkbox[value="${symptomKey}"]`);
        if (cb) cb.checked = true;
      }
      if (temperatureInput) temperatureInput.value = "";
      updateCounter();
    });
  });

  // ---------------- Dark mode toggle (persisted in localStorage) ----------------
  const themeToggleBtn = document.getElementById("theme-toggle");
  const htmlEl = document.documentElement;

  function applyTheme(theme) {
    htmlEl.setAttribute("data-theme", theme);
    if (themeToggleBtn) {
      themeToggleBtn.textContent = theme === "dark" ? "☀️" : "🌙";
    }
  }

  let savedTheme = "light";
  try {
    savedTheme = localStorage.getItem("medisense-theme") || "light";
  } catch (e) {
    // localStorage unavailable (e.g. private browsing restrictions) — default to light
  }
  applyTheme(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", function () {
      const current = htmlEl.getAttribute("data-theme") === "dark" ? "dark" : "light";
      const next = current === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        localStorage.setItem("medisense-theme", next);
      } catch (e) {
        // ignore if storage is unavailable
      }
    });
  }
});
