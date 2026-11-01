# Label guidelines for the RQ3 blind audit

_Generated from the MIT AI Risk Repository taxonomy text stored in the AIID snapshot (taxa record `MIT`, fields `Risk Domain` and `Risk Subdomain`); nothing here is paraphrased. Classifications in AIID using this taxonomy were generated automatically (as stated in the taxonomy record); you are asked for an independent human judgement._

## Your task

For each of the 150 incidents in `data/manual/label_audit_sample.csv`, read the title and description and choose the **one** risk domain that best describes the harm. Do **not** look up the existing MIT label for these incidents (the audit is blind); the sample is shuffled and the label is not in the file.

- `my_label`: one of the seven domain names below (exact name, with or without its number, or just the digit 1-7).
- `my_confidence`: 1 = low, 2 = medium, 3 = high.
- `notes`: optional (for example, which two domains you hesitated between).
- Judge the primary harm described, not the technology; if two domains fit, pick the one that is more central and say so in `notes`.
- Alternatively use the helper: `uv run streamlit run scripts/label_audit_app.py` (it saves the same CSV).

## Domains (allowed values of `my_label`)

1. Discrimination and Toxicity
2. Privacy & Security
3. Misinformation
4. Malicious Actors & Misuse
5. Human-Computer Interaction
6. Socioeconomic & Environmental Harms
7. AI system safety, failures, and limitations

## Domain description (MIT, verbatim)

The Domain Taxonomy of AI Risks classifies risks into seven AI risk domains: (1) Discrimination & toxicity, (2) Privacy & security, (3) Misinformation, (4) Malicious actors & misuse, (5) Human-computer interaction, (6) Socioeconomic & environmental harms, and (7) AI system safety, failures & limitations.

## Subdomain definitions (MIT, verbatim)

# 1 Discrimination & toxicity

## 1.1 Unfair discrimination and misrepresentation

Unequal treatment of individuals or groups by AI, often based on race, gender, or other sensitive characteristics, resulting in unfair outcomes and unfair representation of those groups.

## 1.2 Exposure to toxic content 

AI that exposes users to harmful, abusive, unsafe or inappropriate content. May involve providing advice or encouraging action. Examples of toxic content include hate speech, violence, extremism, illegal acts, or child sexual abuse material, as well as content that violates community norms such as profanity, inﬂammatory political speech, or pornography.

## 1.3 Unequal performance across groups

Accuracy and effectiveness of AI decisions and actions is dependent on group membership, where decisions in AI system design and biased training data lead to unequal outcomes, reduced beneﬁts, increased effort, and alienation of users.

# 2 Privacy & security

## 2.1 Compromise of privacy by obtaining, leaking, or correctly inferring sensitive information

AI systems that memorize and leak sensitive personal data or infer private information about individuals without their consent. Unexpected or unauthorized sharing of data and information can compromise user expectation of privacy, assist identity theft, or cause loss of conﬁdential intellectual property.

## 2.2 AI system security vulnerabilities and attacks

Vulnerabilities that can be exploited in AI systems, software development toolchains, and hardware, resulting in unauthorized access, data and privacy breaches, or system manipulation causing unsafe outputs or behavior.

# 3 Misinformation

## 3.1 False or misleading information

AI systems that inadvertently generate or spread incorrect or deceptive information, which can lead to inaccurate beliefs in users and undermine their autonomy.

Humans that make decisions based on false beliefs can experience physical, emotional, or material harms

## 3.2 Pollution of information ecosystem and loss of consensus reality

Highly personalized AI-generated misinformation that creates “ﬁlter bubbles” where individuals only see what matches their existing beliefs, undermining shared reality and weakening social cohesion and political processes.

# 4 Malicious actors & misuse

## 4.1 Disinformation, surveillance, and inﬂuence at scale

Using AI systems to conduct large-scale disinformation campaigns, malicious surveillance, or targeted and sophisticated automated censorship and propaganda, with the aim of manipulating political processes, public opinion, and behavior.

## 4.2 Cyberattacks, weapon development or use, and mass harm

Using AI systems to develop cyber weapons (e.g., by coding cheaper, more effective malware), develop new or enhance existing weapons (e.g., Lethal Autonomous Weapons or chemical, biological, radiological, nuclear, and high-yield explosives), or use weapons to cause mass harm.

## 4.3 Fraud, scams, and targeted manipulation

Using AI systems to gain a personal advantage over others such as through cheating, fraud, scams, blackmail, or targeted manipulation of beliefs or behavior. Examples include AI-facilitated plagiarism for research or education, impersonating a trusted or fake individual for illegitimate ﬁnancial beneﬁt, or creating humiliating or sexual imagery.

# 5 Human-computer interaction

## 5.1 Overreliance and unsafe use

Anthropomorphizing, trusting, or relying on AI systems by users, leading to emotional or material dependence and to inappropriate relationships with or expectations of AI systems. Trust can be exploited by malicious actors (e.g., to harvest information or enable manipulation), or result in harm from inappropriate use of AI in critical situations (e.g., medical emergency). Over reliance on AI systems can compromise autonomy and weaken social ties

## 5.2 Loss of human agency and autonomy

Delegating by humans of key decisions to AI systems, or AI systems that make decisions that diminish human control and autonomy, potentially leading to humans feeling disempowered, losing the ability to shape a fulfilling life trajectory, or becoming cognitively enfeebled.

# 6 Socioeconomic & environmental harms

## 6.1 Power centralization and unfair distribution of benefits

AI-driven concentration of power and resources within certain entities or groups, especially those with access to or ownership of powerful AI systems, leading to inequitable distribution of benefits and increased societal inequality.

## 6.2 Increased inequality and decline in employment quality

Social and economic inequalities caused by widespread use of AI, such as by automating jobs, reducing the quality of employment, or producing exploitative dependencies between workers and their employers.

## 6.3 Economic and cultural devaluation of human effort

AI systems capable of creating economic or cultural value, including through reproduction of human innovation or creativity (e.g., art, music, writing, coding, invention), destabilizing economic and social systems that rely on human effort. The ubiquity of AI-generated content may lead to reduced appreciation for human skills, disruption of creative and knowledge-based industries, and homogenization of cultural experiences.

## 6.4 Competitive dynamics 

Competition by AI developers or state-like actors in an AI “race” by rapidly developing, deploying, and applying AI systems to maximize strategic or economic advantage, increasing the risk they release unsafe and error-prone systems.

## 6.5 Governance failure 

Inadequate regulatory frameworks and oversight mechanisms that fail to keep pace with AI development, leading to ineffective governance and the inability to manage AI risks appropriately.

## 6.6 Environmental harm

The development and operation of AI systems that cause environmental harm, such as through energy consumption of data centers or the materials and carbon footprints associated with AI hardware.

# 7 AI system safety, failures & limitations

## 7.1 AI pursuing its own goals in conflict with human goals or values

AI systems that act in conflict with ethical standards or human goals or values, especially the goals of designers or users. These misaligned behaviors may be introduced by humans during design and development, such as through reward hacking and goal misgeneralisation, and may result in AI using dangerous
capabilities such as manipulation, deception, or situational awareness to seek power, self-proliferate, or achieve other goals.

## 7.2 AI possessing dangerous capabilities

AI systems that develop, access, or are provided with capabilities that increase their potential to cause mass harm through deception, weapons development and acquisition, persuasion and manipulation, political strategy, cyber-offense, AI development, situational awareness, and self-proliferation. These capabilities may cause mass harm due to malicious human actors, misaligned AI systems, or failure in the AI system.

## 7.3 Lack of capability or robustness 

AI systems that fail to perform reliably or effectively under varying conditions, exposing them to errors and failures that can have significant consequences, especially in critical applications or areas that require moral reasoning.

## 7.4 Lack of transparency or interpretability

Challenges in understanding or explaining the decision-making processes of AI systems, which can lead to mistrust, difficulty in enforcing compliance standards or holding relevant actors accountable for harms, and the inability to identify and correct errors.

## 7.5 AI welfare and rights 

Ethical considerations regarding the treatment of potentially sentient AI entities, including discussions around their potential rights and welfare, particularly as AI systems become more advanced and autonomous.

## Source

Taxonomy record description: [The MIT AI Risk Repository](https://airisk.mit.edu/) contains detailed records of AI-related risks extracted from a variety of sources, categorized into high-level and mid-level taxonomies. Its high-level Causal Taxonomy includes attributes such as the entity responsible for the risk (human, AI, or other), the intent (intentional, unintentional, or other), and the timing (pre-deployment, post-deployment, or other). Its mid-level Domain Taxonomy categorizes risks into 23 specific domains like discrimination, misinformation, malicious use, and human-computer interaction issues.

Classifications using this taxonomy were generated automatically through the work of [Simon Mylius](https://simonmylius.com/incident-classification/introduction). This taxonomy description is provisional and adapted from "[The AI Risk Repository: A Comprehensive Meta-Review, Database, and Taxonomy of Risks From Artificial Intelligence](https://arxiv.org/pdf/2408.12622)."

