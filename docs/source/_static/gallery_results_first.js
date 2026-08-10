/*
 * Give Sphinx-Gallery detail pages a result-first reading order without
 * changing or hiding the generated scientific source.  Figures and terminal
 * output are moved into a compact preview after the example introduction;
 * the complete code and download links remain below.
 */
(function () {
  "use strict";

  function directHeading(section) {
    for (const child of section.children) {
      if (/^H[2-6]$/.test(child.tagName)) {
        return child.textContent.replace("", "").trim();
      }
    }
    return "";
  }

  function resultsFirst() {
    const example = document.querySelector("section.sphx-glr-example-title");
    if (!example || example.querySelector(":scope > .otter-results-first")) {
      return;
    }

    const images = Array.from(
      example.querySelectorAll(
        "img.sphx-glr-single-img, img.sphx-glr-multi-img"
      )
    );
    const outputs = Array.from(
      example.querySelectorAll(".sphx-glr-script-out")
    );
    if (images.length === 0 && outputs.length === 0) {
      return;
    }

    const preview = document.createElement("section");
    preview.className = "otter-results-first";
    preview.setAttribute("aria-label", "Calculated results");

    const title = document.createElement("h2");
    title.textContent = "Calculated results";
    preview.appendChild(title);

    if (images.length > 0) {
      const grid = document.createElement("div");
      grid.className = "otter-result-grid";

      for (const image of images) {
        const sourceSection = image.closest("section");
        const captionText = sourceSection ? directHeading(sourceSection) : "";
        const card = document.createElement("figure");
        card.className = "otter-result-card";
        card.appendChild(image);

        if (captionText) {
          const caption = document.createElement("figcaption");
          caption.textContent = captionText;
          card.appendChild(caption);
        }
        grid.appendChild(card);
      }
      preview.appendChild(grid);
    }

    if (outputs.length > 0) {
      const terminal = document.createElement("div");
      terminal.className = "otter-terminal-results";
      const terminalTitle = document.createElement("h3");
      terminalTitle.textContent = "Terminal output";
      terminal.appendChild(terminalTitle);
      for (const output of outputs) {
        terminal.appendChild(output);
      }
      preview.appendChild(terminal);
    }

    const firstCode = Array.from(example.children).find(function (child) {
      return child.matches(
        ".highlight-Python, .highlight-python, .highlight-default"
      );
    });
    if (firstCode) {
      example.insertBefore(preview, firstCode);
    } else {
      example.appendChild(preview);
    }
  }

  function annotateExampleGallery() {
    const gallery = document.querySelector(
      "section#example-gallery > .sphx-glr-thumbnails"
    );
    if (!gallery || gallery.previousElementSibling?.classList.contains(
      "otter-gallery-run-note"
    )) {
      return;
    }
    const note = document.createElement("p");
    note.className = "otter-gallery-run-note";
    note.textContent =
      "Each page includes a complete, directly executable Python script " +
      "and a source download.";
    gallery.parentNode.insertBefore(note, gallery);
  }

  function initialiseGalleryPresentation() {
    annotateExampleGallery();
    resultsFirst();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialiseGalleryPresentation);
  } else {
    initialiseGalleryPresentation();
  }
})();
