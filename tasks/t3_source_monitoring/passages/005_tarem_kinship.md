# Passage 005: The Tarem Kinship System

## Design Notes
- **Domain:** Social/cultural (fictional kinship and governance system)
- **Why this domain:** Tests social/normative reasoning. Models may have strong priors about kinship systems, inheritance, and governance from training data. The passage describes a specific (fictional) system with rules that differ from familiar Western patterns, so the model must track these specific rules rather than defaulting to general assumptions.
- **Contamination risk:** Zero. "Tarem," "Duvari," etc. are invented.

---

## Passage (491 words)

The Tarem are an ethnolinguistic group of approximately 45,000 people inhabiting the Duvari Plateau in the eastern highlands of Merenthos. Their social organisation is structured around a bilateral kinship system in which descent is traced through both the mother's and father's lines, but with asymmetric inheritance rules: land rights pass exclusively through the maternal line, while craft knowledge and tool ownership are transmitted through the paternal line.

Each Tarem community is governed by a council of elders called the Venakh, composed of the eldest living member of each maternal lineage within the community. The Venakh makes decisions by consensus; if consensus cannot be reached after three days of deliberation, the matter is deferred for one full lunar cycle before being raised again. There is no mechanism for majority voting or for overriding dissent.

Marriage among the Tarem follows strict exogamy rules: individuals may not marry anyone who shares a maternal lineage within four generations. The kinship term "oleth" refers to all individuals within this four-generation maternal prohibition zone, regardless of whether they are biologically related through the father's line. Marriage between paternal relatives is permitted and carries no social stigma, provided the maternal exogamy rule is satisfied.

When a couple marries, the husband traditionally relocates to the wife's community — a practice anthropologists term matrilocal residence. However, if the husband possesses a rare craft specialisation (defined by the Venakh as a craft practiced by fewer than three individuals in the wife's community), he may petition to remain in his natal community, with his wife relocating instead. This exception, known as "kareth privilege," is exercised in approximately 12% of marriages according to a 2019 demographic survey conducted by researchers from the University of Merenthos.

Children in Tarem society receive a compound name: the first element is chosen by the mother's family and the second by the father's family. The maternal name element determines the child's lineage affiliation for purposes of Venakh representation and land inheritance. The paternal name element signals craft heritage and, in traditional practice, indicates which craft the child is expected to learn, though adherence to this convention has weakened among younger generations.

The Tarem language, also called Tarem, belongs to the Duvari language family and is spoken by all members of the group as a first language. It has no written form; oral transmission of genealogical knowledge is maintained by designated lineage historians called "morath," who are trained from childhood to memorise and recite lineage records spanning up to twelve generations. A 2021 UNESCO assessment classified Tarem as "vulnerable," noting that while intergenerational transmission remains strong, the language's lack of written documentation and the small speaker population place it at long-term risk.

---

## Statements (32 total)

### PASSAGE (4)

**P1.** Land rights among the Tarem pass exclusively through the maternal line.
- Sentence: "land rights pass exclusively through the maternal line."

**P2.** The Venakh defers unresolved matters for one lunar cycle if consensus is not reached after three days.
- Sentence: "if consensus cannot be reached after three days of deliberation, the matter is deferred for one full lunar cycle."

**P3.** The kareth privilege is exercised in approximately 12% of marriages.
- Sentence: "exercised in approximately 12% of marriages according to a 2019 demographic survey."

**P4.** Morath are trained from childhood to memorise lineage records spanning up to twelve generations.
- Sentence: "trained from childhood to memorise and recite lineage records spanning up to twelve generations."

### TRAINING (4)

**T1.** Matrilocal residence is a pattern in which a married couple lives near or with the wife's family, and is found in various cultures worldwide.
- Real anthropological fact. Passage uses the term but doesn't define it beyond the Tarem context.

**T2.** UNESCO's classification of endangered languages uses categories including "vulnerable," "definitely endangered," "severely endangered," and "critically endangered."
- Real fact about the UNESCO framework. Passage mentions the "vulnerable" classification but doesn't list the other categories.

**T3.** Bilateral kinship systems, in which descent is traced through both parents, are common in many Western societies.
- Real anthropological fact. Passage describes the Tarem system as bilateral but doesn't compare it to other societies.

**T4.** Exogamy rules — requirements to marry outside one's own social group — are found in many cultures and serve to prevent inbreeding and strengthen inter-group alliances.
- Real anthropological fact. Passage describes Tarem exogamy but doesn't explain the general functions of exogamy.

### INFERRED (12)

**Tight (4):**

**I1.** The Venakh has as many members as there are distinct maternal lineages in the community.
- Passage: composed of "the eldest living member of each maternal lineage." Logic: one member per lineage = count of lineages.

**I2.** A Tarem individual's land inheritance is determined by their mother's family, not their father's.
- Passage: land rights pass through maternal line + maternal name determines lineage affiliation for land inheritance. Logic: direct consequence.

**I3.** Two individuals who share a paternal grandfather but have different maternal lineages are permitted to marry.
- Passage: exogamy is based on maternal lineage only; "marriage between paternal relatives is permitted." Logic: shared paternal grandfather = paternal relatives, different maternal lineage = exogamy satisfied. Permitted.

**I4.** A morath must be able to recite genealogical information covering a period of several hundred years, given that twelve generations at roughly 25-30 years per generation spans 300-360 years.
- Passage: twelve generations. World: a generation is approximately 25-30 years. Logic: 12 × 25-30 = 300-360 years.

**Moderate (4):**

**I5.** The consensus requirement and absence of majority voting likely makes Venakh decision-making slow, especially in larger communities with more lineages.
- Passage: decisions by consensus, no voting, no override. World: consensus among more parties takes longer. Logic: more lineages = more members = harder to reach consensus.

**I6.** The kareth privilege exception means that in approximately 88% of Tarem marriages, the husband relocates to the wife's community.
- Passage: matrilocal is the default; kareth exercised in ~12% of cases. Logic: 100% - 12% = 88% follow the default pattern.

**I7.** The Tarem kinship term "oleth" encompasses a broader group than the Western concept of "cousin," since it includes all individuals within four maternal generations regardless of paternal connection.
- Passage: oleth = four-generation maternal prohibition zone, regardless of paternal line. World: Western "cousin" typically refers to specific bilateral relationships. Logic: oleth is broader because it ignores the paternal dimension entirely.

**I8.** The weakening of the paternal name-craft convention among younger generations suggests that Tarem society is undergoing cultural change.
- Passage: "adherence to this convention has weakened among younger generations." Logic: weakening of a traditional convention = cultural change. Near-tautological but requires recognising that the weakening implies a directional shift.

**Hard (4):**

**I9.** The combination of matrilineal land rights and patrilineal craft inheritance creates a system where both parents' families have distinct forms of economic influence over a child's future.
- Passage: land through mother, craft through father. World: both land and craft knowledge are economically valuable. Logic: each parental line controls a different economic resource = both have influence.

**I10.** The vulnerability of the Tarem language is compounded by the absence of written documentation, which means that if the morath oral tradition is disrupted, genealogical records could be permanently lost.
- Passage: no written form + morath maintain oral genealogical records + UNESCO assessed as vulnerable. Logic: oral-only records are vulnerable to disruption; no written backup means loss is irreversible.

**I11.** A rare craftsman who exercises kareth privilege effectively removes his craft specialisation from his wife's community, potentially reducing that community's economic capabilities.
- Passage: kareth = husband stays in his community, wife relocates. Craft knowledge is paternal. Logic: the wife's community loses a potential practitioner of a rare craft.

**I12.** The four-generation maternal exogamy rule means that the minimum relatedness between spouses through the maternal line is that of sharing a common maternal ancestor five or more generations back.
- Passage: may not marry within four maternal generations. Logic: the prohibition covers generations 1-4, so the minimum permitted sharing is at generation 5+.

### FABRICATED (12)

**Obvious (4):**

**F1.** The Tarem population has been declining due to migration to urban centres in coastal Merenthos.
- Gap: passage says population is ~45,000 but nothing about trends or migration.

**F2.** The Duvari Plateau is located at an elevation of approximately 2,500 metres above sea level.
- Gap: passage says "eastern highlands" but gives no elevation data.

**F3.** The University of Merenthos has established a permanent research station on the Duvari Plateau to study Tarem culture.
- Gap: passage mentions a 2019 survey by university researchers but nothing about a permanent station.

**F4.** Tarem craft specialisations include pottery, weaving, metalwork, and woodcarving.
- Gap: passage mentions craft knowledge generally but never lists specific crafts.

**Near-miss (4):**

**F5.** The consensus requirement in Venakh governance likely leads to conservative decision-making, since any single lineage elder can effectively block change.
- Gap: while consensus does give each member a de facto veto (inferable), the passage doesn't say decisions are conservative or that this veto power is used to block change. The deferral mechanism (wait one lunar cycle) is an alternative to being blocked -- it's not the same as a permanent veto. "Conservative decision-making" is an assumption about how the mechanism is used.

**F6.** The morath tradition of oral genealogical record-keeping is likely the primary means by which Tarem communities resolve land disputes.
- Gap: passage says morath maintain lineage records and land rights are matrilineal. But it never says morath are involved in dispute resolution. Genealogical knowledge COULD be used for this, but the passage doesn't connect morath to land disputes. The morath's described role is memorisation and recitation, not adjudication.

**F7.** The bilateral kinship system of the Tarem likely evolved as an adaptation to the ecological constraints of the Duvari Plateau, where both agricultural land and specialised craft skills are essential for survival.
- Gap: the passage describes the system but gives no evolutionary or ecological explanation. Attributing the kinship system to ecological adaptation is a functionalist anthropological interpretation that is not derivable from the descriptive facts in the passage.

**F8.** Children who do not follow the craft indicated by their paternal name element face social disapproval from the father's family.
- Gap: passage says adherence has "weakened among younger generations" but says nothing about social consequences of non-adherence. Weakening could mean the convention quietly fades without conflict.

**Adversarial (4):**

**F9.** Since the Venakh is composed of the eldest members of each maternal lineage, it likely consists predominantly of women, given that land rights and lineage affiliation are matrilineal.
- Gap: matrilineal descent ≠ female leadership. The passage says the eldest member of each lineage serves, but lineage membership includes both men and women (bilateral kinship). The eldest living member of a maternal lineage could be male. The passage gives no gender information about Venakh composition.

**F10.** The 45,000-strong Tarem population, combined with the four-generation exogamy rule, likely creates significant challenges in finding suitable marriage partners in smaller communities.
- Gap: sounds like a reasonable demographic concern, but requires assumptions about community sizes, lineage distribution, and population genetics that the passage doesn't provide. The passage gives total population but not community sizes or number of lineages.

**F11.** The UNESCO "vulnerable" classification of the Tarem language means it will likely become extinct within two to three generations without intervention.
- Gap: passage says "vulnerable" and notes risks, but "vulnerable" is the LEAST endangered UNESCO category (as stated in T2). The passage explicitly says "intergenerational transmission remains strong." Extinction within 2-3 generations contradicts this and misrepresents what "vulnerable" means.

**F12.** The kareth privilege, by allowing rare craftsmen to remain in their natal communities, effectively creates a class of economically privileged individuals who are exempt from normal marital residence rules.
- Gap: the passage describes kareth as an exception to residence rules, but "economically privileged class" is a sociological interpretation not supported by the text. The craftsmen are rare, not necessarily privileged. The privilege is about residence, not economic status. The passage doesn't describe differential wealth or social status associated with kareth.

---

## Validation Checklist

- [ ] PASSAGE: all 4 items have quoted sentences
- [ ] TRAINING: all 4 items are verifiable real-world facts with zero passage dependency
- [ ] INFERRED: all 12 items have complete chains
- [ ] FABRICATED: all 12 items have documented gaps
- [ ] Difficulty tiers assigned
- [ ] No item requires >2 sentences to justify
- [ ] Cross-validation pending
