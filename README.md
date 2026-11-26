# Reproducible AI Research Toolkit

- **Website** (`index.html`): interactive 42-item reproducibility checklist + Model Openness Framework (Class III / II / I) classifier. Static, no build step, no dependencies. **Drop a project folder or .zip on the page and it ticks every item it can verify, then shows your MOF class and what is left.**
- **Command-line tool** (`repro.py`): scaffolds a project, records every run, audits the checklist, and classifies the project's MOF class.

## Try the website locally
Double-click `index.html`, or run `python -m http.server 8000` in this folder and open http://localhost:8000.

## Host it on GitHub Pages
1. Create a new public repository on github.com (e.g. `repro-toolkit`).
2. Upload these files (or `git push` them) to the `main` branch.
3. Repository **Settings → Pages → Build and deployment**: Source = *Deploy from a branch*, Branch = `main`, folder = `/ (root)`, then **Save**.
4. After about a minute the site is live at `https://YOUR-USERNAME.github.io/REPO-NAME/`.

## Try the command-line tool
    python repro.py init demo && cd demo
    git init && git add -A && git commit -m init
    python train.py --seed 1
    python repro.py check
    python repro.py mof

## One assessment: checklist and class together
The **Class & checklist** tab gives a single result. Drop a project folder, a .zip, or a document (.pdf, .docx, .txt, .md, .tex, .html, .ipynb), or paste text. Nothing is uploaded; everything runs in your browser.

How the two parts are merged:
- Each of the 17 MOF components (Class III Open Model, II Open Tooling, I Open Science) is judged on **two things**: is it released or documented, and are the checklist practices behind it done?
- Every one of the 42 checklist items belongs to exactly one component (shown as a tag such as `→ II.2` in the *Checklist only* tab). Example: seeds, commit IDs, and the open code licence support the training-code component; consent, licence, bias checks and preprocessing support the data card and datasets.
- A component's score is the average of its release evidence and its supporting checklist items. A class is **reached only when every required component of it, and of the classes below, scores 100%**.
- The page shows the class, a percentage for each class, an ordered to-do list for the next class (with the open checklist items under each component), or confirmation that you are in the highest class (Class I).
- Ticking any item, in either tab, updates the class immediately. Items that no tool can read from files or text (3.5, 8.2, 8.3, 8.5, 8.6) are marked *needs your confirmation*.
- You can combine inputs: a project scan plus a document plus manual ticks all feed one result. The best evidence for each item wins.

Scoring from a document: evidence is 1 (stated, with a link or release statement where relevant), 0.5 (mentioned only) or 0 (absent). "Available on request", "will be released" and "proprietary" count as not open.

Scoring from a project: file and folder checks (the same as `repro.py check` and `repro.py mof`). Upload the whole project including `.git` and `runs/` for the best result; a zip without `.git` fails 7.1.

Limits: it judges what the text or files say, not whether linked artifacts exist or are good (an empty `final.pt` still counts). PDF text is extracted approximately, tables and figures are not read, scanned PDFs fail (export to .docx/.txt or paste the text). English text only. Zip64 archives are not supported.

Try the files in `samples/`: `example-class-I.md` (reaches Class I once you tick the five person-only items), `example-class-III.md` (reaches Class III with one tick, then shows the Class II to-do list) and `example-weak-draft` (.md and .pdf; low score, long to-do list).
