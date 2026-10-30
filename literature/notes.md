# Literature notes (working file)

Status key: V = checked against the source page in this session, P = partly checked, U = unverified.
Nothing here is copied from a paper. All wording is paraphrased.

## Models: how each does region grounding and finding grounding

| Model | Region (anatomical) grounding | Finding / phrase grounding | Same model does both | Output format and coordinates | Status |
|---|---|---|---|---|---|
| RadVLM | Trained on Chest ImaGenome region boxes, one randomly chosen region per image per training pair (80,000 train, 2,000 eval pairs). Template "Where is the [region] located on the image?" | Abnormality grounding on VinDr-CXR, phrase grounding on MS-CXR (template "Mark the area where you observe: [phrase]") | Yes | Text, [x1, y1, x2, y2] normalised 0 to 1 | V (arXiv HTML, figures 1 and 5) |
| MAIRA-2 | Not trained for anatomical regions (brief); we query phrase grounding with the region name | Grounded report generation and phrase grounding | Region: no | Boxes as tokens, coordinates relative to a square centre crop | P |
| CheXagent | No dedicated region task found in the model card or paper | Phrase grounding, abnormality detection (CheXinstruct fine-grained perception tasks); official helper prompt "Please locate the following phrase: ..." | Not for regions | Text with box tags, scale UNVERIFIED (see conflict below) | P |
| MedGemma 1.5 4B | Anatomical localisation is a stated new capability; Chest ImaGenome used in the reinforcement learning stage (39,968 examples); evaluated on 10,000 Chest ImaGenome examples, mean IoU 38.0 | Not described in the pages read | Unknown | JSON list, box_2d as [y0, x0, y1, x1] | P (arXiv HTML, summarised) |
| ChEX | Prompted detector, text prompts can name anatomical regions | Pathology detection and sentence grounding | Yes | Normalised cx, cy, w, h on a centre-cropped square (from our own script) | P |
| BiomedParse | Only left lung and right lung among our regions; everything else is zero shot | Lung opacity, COVID-19 infection, viral pneumonia on chest X-ray | Partly | Segmentation mask, box derived by us | P (brief) |
| Grounding DINO | Open-set detector, no medical training | Same | n/a | Boxes | V (abstract) |
| BioViL-T | Not a region model | Phrase grounding on MS-CXR via image-text alignment | n/a | Heatmap, box derived by us | P (abstract) |

## Conflicts and open checks

1. CheXagent coordinate scale: a summary of the official helper code divides values by 100, but our Phase 1 outputs range from 145 to 878, which fits a 0 to 999 grid. Phase 1 parser is kept. Read the helper source directly before writing this in the paper.
2. MedGemma 1.5 report text, as summarised, says boxes are normalised 0 to 1. Phase 1 verified a 0 to 1000 grid on real outputs. Keep the verified parser. Read the report section directly.
3. MedGemma abstract reports a 35 percent IoU gain, Table 4 shows 3.1 to 38.0. The report does not reconcile these in the text read. Do not quote the 35 percent figure.
4. BiomedParse journal year: brief said 2024, arXiv page gives Nat Methods 22, 166 to 176 (2025).
5. Self-reported MedGemma IoU 38.0 and RadVLM mAP 85.3 are not comparable to our metrics. Different split, different metric.

## Contamination facts collected so far

- RadVLM: Chest ImaGenome in training. The PhysioNet release (719,675 samples) includes the 80,000 anatomical grounding samples and omits CheXpert and PadChest samples. No statement found about excluding the gold subset.
- MedGemma 1.5: Chest ImaGenome in the reinforcement learning stage. No statement found about excluding gold patients.
- Chest ImaGenome: the official page lists images_to_avoid.csv for gold patients and says mediastinum was derived from upper mediastinum and cardiac silhouette, not annotated directly.

## To research next

MS-CXR, VinDr-CXR, PadChest-GR, CheXOne, MedRG, ViviMed, the original Chest ImaGenome paper, neutral benchmark papers (contamination, prompt sensitivity, laterality and flip tests in medical VLMs), IoU versus mAP reporting conventions, MedGemma 1.5 report details beyond the abstract, RadVLM comparison tables (CheXagent 6.2 / 26.0 / 69.7, MAIRA-2 19.8 / 11.3 / 80.1 as given in the brief, not yet checked).
