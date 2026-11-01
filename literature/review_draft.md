# Related work draft (for supervisor review)

Status: first draft written from sources checked in this session. Every claim that rests only on the project brief, or on a search summary I could not confirm against the paper itself, is tagged UNVERIFIED. Citation keys refer to `bibliography.bib`. The framing is a review of benchmarking papers and an independent benchmarking study, not a claim of publication.

## 2.1 Anatomical region grounding

Chest ImaGenome is the common source of region supervision in this literature. It provides scene graphs for 242,072 MIMIC-CXR images, with bounding boxes for 29 anatomical locations produced by a rule-based language pipeline together with an atlas-based box detector, and a manually corrected gold set from 500 patients \cite{wu2021chestimagenome}. The counts come from a search summary of the paper and are UNVERIFIED against the paper itself.

RadVLM treats region grounding as one of several instruction tasks. Its authors built over one million image-instruction pairs and report visual grounding among the covered tasks \cite{deperrois2025radvlm}. For anatomical grounding, each training pair names one region, chosen at random for the image, and asks where it is; the answer is a box given as top-left and bottom-right corners normalised to the range 0 to 1. The task has 80,000 training pairs and 2,000 evaluation pairs, all drawn from Chest ImaGenome \cite{deperrois2025radvlm}. The released instruction dataset contains the same 80,000 anatomical samples and omits sources covered by other data use terms \cite{radvlminstruction2025}.

MedGemma 1.5 lists anatomical localisation with bounding boxes as a new capability. Its technical report states that Chest ImaGenome contributes 39,968 examples to the reinforcement learning stage, evaluates localisation on 10,000 Chest ImaGenome examples with a mean IoU of 38.0 for the 4B model, and returns boxes as a JSON list in y-first order \cite{sellergren2026medgemma15}. The report does not say which split the 10,000 examples come from, and the text I read does not say whether the gold patients were excluded from training.

ChEX approaches the task with a prompted detector. A text prompt can name an anatomical region or a pathology, and the model returns a box for it \cite{muller2024chex}. AnatomiX uses two stages: it first identifies anatomical structures and extracts their features, then passes them to a language model, and its authors report an improvement of more than 25 percent on anatomy grounding over existing approaches \cite{hashmi2026anatomix}. BiomedParse is a text-prompted segmentation model covering 82 object types across nine imaging modalities \cite{zhao2025biomedparse}. Which chest X-ray structures it was trained on is UNVERIFIED in my reading; the project brief states that, among our regions, only the left and right lung were seen in training.

## 2.2 Finding and phrase grounding

Finding grounding asks for the location of an abnormality or of the finding described by a phrase. RadVLM covers abnormality grounding with templates that name the abnormality and ask for its location, and phrase grounding with a template that asks the model to mark the area described by a phrase \cite{deperrois2025radvlm}. Its instruction data draw on VinDr-CXR for abnormalities and on MS-CXR for phrases \cite{radvlminstruction2025,nguyen2022vindrcxr,boecking2022mscxr}. VinDr-CXR released 18,000 posteroanterior scans with 22 local and 6 global labels \cite{nguyen2022vindrcxr}. MS-CXR holds 1,162 image-sentence pairs with radiologist boxes over eight findings \cite{boecking2022mscxr}.

CheXagent was trained on CheXinstruct, whose fine-grained perception category includes abnormality detection and abnormality grounding, and it was evaluated on phrase grounding with a reported mean IoU of 0.627 and a mean average precision of 0.810 \cite{chen2024chexagent}. The second figure pair comes from a summary of the full text and is UNVERIFIED. For grounding, the authors report giving an open-ended instruction and scoring the boxes in the reply; the model card and the paper give no grounding template, while the official code uses a short instruction asking the model to locate a phrase.

MAIRA-2 generates radiology reports in which findings are tied to boxes on the image, a task its authors call grounded report generation, and it introduces RadFact for evaluating grounded reports \cite{bannur2024maira2}. How boxes are encoded and its MS-CXR result are UNVERIFIED here. BioViL-T adds prior images and reports to vision-language pretraining and reports phrase grounding among its results \cite{bannur2023biovilt}. Grounding DINO is a general-domain open-set detector that takes category names or referring expressions \cite{liu2023groundingdino}, and CheXOne, a 2026 reasoning model, lists visual grounding among its zero-shot evaluations \cite{zhang2026chexone}.

## 2.3 Models that do both

Table 1 summarises which models cover regions, findings or both, as far as the sources I read state. RadVLM is explicit about both, with separate instruction templates and separate source datasets. MedGemma 1.5 describes localisation of anatomical features and findings on its model card. ChEX names regions and pathologies in the same prompt interface. MAIRA-2 and CheXagent are documented for findings and phrases; I did not find a region task for either, so we query them with the region name as a phrase and treat region grounding as outside their trained task.

| Model | Region grounding | Finding or phrase grounding | Output | Status |
|---|---|---|---|---|
| RadVLM | Trained, Chest ImaGenome | Trained, VinDr-CXR and MS-CXR | Text box, normalised 0 to 1 | Checked |
| MedGemma 1.5 | Stated capability, Chest ImaGenome in RL stage | Stated capability | JSON, y-first | Checked in part |
| ChEX | Prompted, trained | Prompted, trained | Box regression | Abstract only |
| AnatomiX | Trained, two-stage | Trained | Not read | Abstract only |
| MAIRA-2 | Not trained for regions (brief) | Grounded reports | Box tokens | UNVERIFIED |
| CheXagent | None found | Phrase and abnormality grounding | Box in text | Checked in part |
| BioViL-T | Not a region model | Phrase grounding | Alignment map | Abstract only |
| Grounding DINO | General-domain | General-domain | Boxes | Abstract only |
| BiomedParse | Lung only (brief) | Some findings | Mask | UNVERIFIED |

## 2.4 How the authors evaluate, and why the numbers do not compare

The published comparisons are self-benchmarks. RadVLM reports mean average precision on a Chest ImaGenome test set, and its authors attribute part of the gap to other models to Chest ImaGenome being in RadVLM's training data (UNVERIFIED, from the brief). MedGemma 1.5 reports mean IoU on 10,000 Chest ImaGenome examples \cite{sellergren2026medgemma15}. One uses a detection metric and the other an overlap metric, on different subsets and with different prompts, so the figures cannot be placed in one table. Mean IoU and mean average precision at a threshold measure different things, and we report both.

## 2.5 Neutral evaluation work and the gap

Work on evaluation outside self-benchmarks is thin in what I found. A paraphrase-sensitivity study of chest X-ray question answering reports answer flips of 8 to 58 percent across six medical models \cite{psfmed2026}, which concerns questions rather than boxes. A benchmark of localisation for contrastive models reports weaker results on small or diffuse lesions \cite{xbench2025}. A grounding-data paper notes left and right errors in its own verification step \cite{medground2026}. All three are UNVERIFIED beyond a search summary.

Within the sources I checked, I found no study that compares anatomical region grounding across these models under one protocol, audits overlap between each model's training data and the evaluation images, or tests left and right behaviour by mirroring the images. AnatomiX is reported to test flipped images, which I could not confirm from its abstract and will read in full. Any statement of the gap must therefore be limited to the sources checked.
