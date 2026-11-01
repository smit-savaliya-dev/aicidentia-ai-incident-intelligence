| group | column | kind | coverage_n | coverage_pct | value_type | share_multi | n_classes | top_class | top_share | min_class_n | imbalance_ratio | norm_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MIT | Risk Domain | categorical | 1496 | 0.843 | single-valued | 0 | 7 | Malicious Actors & Misuse | 0.378 | 24 | 23.6 | 0.831 |
| MIT | Risk Subdomain | categorical | 1495 | 0.842 | single-valued | 0 | 21 | Fraud, scams, and targeted manipulation | 0.276 | 1 | 412 | 0.706 |
| MIT | Responsible Entity | categorical | 1496 | 0.843 | single-valued | 0 | 3 | AI | 0.642 | 11 | 87.3 | 0.626 |
| MIT | Intent | categorical | 1496 | 0.843 | single-valued | 0 | 3 | Unintentional | 0.495 | 21 | 35.2 | 0.689 |
| MIT | Timing | categorical | 1496 | 0.843 | single-valued | 0 | 3 | Post-deployment | 0.976 | 9 | 162 | 0.116 |
| GMF | AI Goal | categorical | 41 | 0.0231 | multi-valued | 0.122 | 35 | Autonomous Driving | 0.0816 | 1 | 4 | 0.966 |
| GMF | AI Technology | categorical | 26 | 0.0146 | multi-valued | 0.654 | 25 | Distributional Learning | 0.16 | 1 | 8 | 0.914 |
| GMF | AI Technology (Potential) | categorical | 29 | 0.0163 | multi-valued | 0.517 | 22 | Regression | 0.12 | 1 | 6 | 0.929 |
| GMF | Technical Failure | categorical | 26 | 0.0146 | multi-valued | 0.5 | 26 | Distributional Bias | 0.122 | 1 | 6 | 0.936 |
| GMF | Technical Failure (Potential) | categorical | 36 | 0.0203 | multi-valued | 0.806 | 36 | Generalization Failure | 0.104 | 1 | 11 | 0.927 |
| CSETv1 | Harm Domain | categorical | 200 | 0.113 | single-valued | 0 | 3 | yes | 0.79 | 13 | 12.2 | 0.586 |
| CSETv1 | Tangible Harm | categorical | 200 | 0.113 | single-valued | 0 | 5 | no tangible harm, near-miss, or issue | 0.465 | 3 | 31 | 0.739 |
| CSETv1 | AI Harm Level | categorical | 200 | 0.113 | single-valued | 0 | 5 | none | 0.63 | 10 | 12.6 | 0.667 |
| CSETv1 | Rights Violation | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.77 | 18 | 8.56 | 0.631 |
| CSETv1 | Harm Distribution Basis | categorical | 197 | 0.111 | multi-valued | 0.183 | 14 | none | 0.455 | 1 | 122 | 0.715 |
| CSETv1 | Lives Lost | numeric | 206 | 0.116 | single-valued | 0 |  |  |  |  |  |  |
| CSETv1 | Injuries | numeric | 206 | 0.116 | single-valued | 0 |  |  |  |  |  |  |
| CSETv1 | Sector of Deployment | categorical | 200 | 0.113 | multi-valued | 0.315 | 18 | information and communication | 0.303 | 1 | 81 | 0.807 |
| CSETv1 | Public Sector Deployment | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.78 | 3 | 52 | 0.529 |
| CSETv1 | Infrastructure Sectors | categorical | 54 | 0.0304 | multi-valued | 0.0185 | 12 | transportation | 0.255 | 1 | 14 | 0.866 |
| CSETv1 | Location Region | categorical | 200 | 0.113 | single-valued | 0 | 7 | North America | 0.475 | 1 | 95 | 0.682 |
| CSETv1 | Country Code | categorical | 140 | 0.0789 | single-valued | 0 | 26 | US | 0.65 | 1 | 91 | 0.525 |
| CSETv1 | Location City | categorical | 54 | 0.0304 | single-valued | 0 | 42 | Los Angeles | 0.0926 | 1 | 5 | 0.965 |
| CSETv1 | Incident Year | numeric | 200 | 0.113 | single-valued | 0 |  |  |  |  |  |  |
| CSETv1 | Incident Month | categorical | 143 | 0.0806 | single-valued | 0 | 14 | December | 0.126 | 1 | 18 | 0.926 |
| CSETv1 | Incident Day | categorical | 74 | 0.0417 | single-valued | 0 | 29 | 29 | 0.0676 | 1 | 5 | 0.959 |
| CSETv1 | Intentional Harm | categorical | 198 | 0.112 | single-valued | 0 | 4 | No. Not intentionally designed to perform harm | 0.919 | 1 | 182 | 0.254 |
| CSETv1 | Autonomy Level | categorical | 200 | 0.113 | single-valued | 0 | 4 | Autonomy1 | 0.52 | 16 | 6.5 | 0.84 |
| CSETv1 | AI Task | free_text | 168 | 0.0946 | multi-valued | 0.518 |  |  |  |  |  |  |
| CSETv1 | AI Methods | categorical | 91 | 0.0513 | multi-valued | 0.472 | 67 | computer vision | 0.131 | 1 | 20 | 0.872 |
| CSETv1 | AI System Description | free_text | 196 | 0.11 | single-valued | 0 |  |  |  |  |  |  |
| CSETv1 | Data Inputs | categorical | 193 | 0.109 | multi-valued | 0.751 | 254 | text | 0.0765 | 1 | 40 | 0.888 |
| CSETv1 | Involving Minor | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.89 | 4 | 44.5 | 0.363 |
| CSETv1 | Protected Characteristic | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.635 | 4 | 31.8 | 0.668 |
| CSETv1 | AI System | categorical | 200 | 0.113 | single-valued | 0 | 3 | yes | 0.805 | 11 | 14.6 | 0.555 |
| CSETv1 | Detrimental Content | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.77 | 7 | 22 | 0.58 |
| CSETv1 | Impact on Critical Services | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.9 | 2 | 90 | 0.326 |
| CSETv1 | Multiple AI Interaction | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.97 | 1 | 194 | 0.135 |
| CSETv1 | Embedded | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.725 | 3 | 48.3 | 0.588 |
| CSETv1 | Entities | free_text | 200 | 0.113 | multi-valued | 1 |  |  |  |  |  |  |
| CSETv1 | Physical Objects | categorical | 200 | 0.113 | single-valued | 0 | 3 | no | 0.68 | 2 | 68 | 0.611 |
| CSETv1 | Deployed | categorical | 200 | 0.113 | single-valued | 0 | 3 | yes | 0.925 | 3 | 61.7 | 0.277 |
| CSETv1 | Estimated Date | categorical | 207 | 0.117 | single-valued | 0 | 2 | False | 0.667 | 69 | 2 | 0.918 |
| CSETv1 | Estimated Harm Quantities | categorical | 207 | 0.117 | single-valued | 0 | 2 | False | 0.927 | 15 | 12.8 | 0.375 |
| CSETv1 | Special Interest Intangible Harm | categorical | 200 | 0.113 | single-valued | 0 | 3 | yes | 0.515 | 10 | 10.3 | 0.777 |
