const languageButtons = document.querySelectorAll("[data-language]");
const languageContents = document.querySelectorAll("[data-content]");

function setLanguage(language) {
  const selectedLanguage = document.querySelector(".language-content:not([hidden])")?.dataset.content;

  document.documentElement.lang = language === "zh" ? "zh-Hans" : "en";

  for (const button of languageButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  }

  for (const content of languageContents) {
    content.hidden = content.dataset.content !== language;
  }

  if (selectedLanguage !== language) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

for (const button of languageButtons) {
  button.addEventListener("click", () => setLanguage(button.dataset.language));
}
