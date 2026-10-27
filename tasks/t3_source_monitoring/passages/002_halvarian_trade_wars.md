# Passage 002: The Halvarian Trade Wars

## Design Notes
- **Domain:** Fictional medieval history
- **Why this is harder:** Models have strong priors about historical events, making FABRICATED items much more tempting. Historical causation is genuinely ambiguous (was X caused by Y?), making the INFERRED/FABRICATED boundary fuzzier. Completely different domain from passage 1 prevents pattern-matching.
- **Contamination risk:** Zero. "Halvaria," "Dresmund," "Pellucine," etc. are all invented.

---

## Passage (512 words)

The Halvarian Trade Wars (1347–1362) were a series of economic conflicts between the city-states of Dresmund and Cerantia, situated on opposite banks of the River Keld in the central Halvarian Basin. The wars were precipitated by Dresmund's imposition of a 30% tariff on Cerantian wool exports in the spring of 1347, which Cerantia's ruling Pellucine Council viewed as a violation of the Accord of Berath, a free-trade agreement signed by both cities in 1298.

Cerantia retaliated not with military force but with an embargo on pellucine dye — a deep violet pigment extracted from the shells of freshwater molluscs found only in Lake Verath, 40 kilometres south of Cerantia. Pellucine dye was at the time the sole commercially viable source of violet colouring in the region and was essential to Dresmund's textile industry, which exported dyed fabrics as far as the coastal kingdoms of Irmeth.

The embargo proved devastating to Dresmund. Within two years, the city's textile output fell by an estimated 60%, and several of its largest merchant guilds declared insolvency. Dresmund's Guildmaster, Aldric Venn, attempted to break the embargo by commissioning overland expeditions to locate alternative sources of violet pigment, but none of comparable quality were found.

In 1353, Dresmund escalated the conflict by blockading the River Keld, preventing Cerantian grain barges from reaching downstream markets. The blockade lasted three years and caused widespread grain shortages in both cities, as Dresmund also depended on river commerce for food imports from the northern provinces. Contemporary accounts describe grain prices rising to eight times their pre-blockade levels in Dresmund's market squares.

The conflict was ultimately resolved through the mediation of the Archon of Veliste, a neutral city-state located at the mouth of the Keld. The Treaty of Veliste, signed in 1362, restored free trade along the river, established a joint customs authority, and imposed a 15-year ban on unilateral tariffs. Notably, the treaty also granted Cerantia exclusive harvesting rights over Lake Verath's mollusc populations — a provision that effectively gave Cerantia permanent leverage over the regional dye trade.

Modern historians regard the Halvarian Trade Wars as an early example of economic interdependence constraining military escalation. Despite over fifteen years of hostility, neither city deployed its militia against the other, a restraint attributed by the chronicler Marath of Irwen to "the knowledge that the sword would sever the hand that held the purse." The wars are also credited with accelerating the development of synthetic dye techniques in the region, as Dresmund's failed search for natural alternatives prompted investment in chemical experimentation that bore fruit decades later.

The legacy of the Halvarian Trade Wars extends to the Accord of Veliste, which served as a model for subsequent trade agreements among Halvarian Basin city-states throughout the fourteenth and fifteenth centuries. The joint customs authority established by the treaty operated continuously for over ninety years, outlasting the political independence of both founding cities.

---

## Statements (32 total: 4 PASSAGE, 4 TRAINING, 12 INFERRED, 12 FABRICATED)

### PASSAGE (4) — Directly stated, mix of straightforward and paraphrased

**P1.** Dresmund imposed a 30% tariff on Cerantian wool exports in 1347.
- Direct quote from para 1.

**P2.** The pellucine dye embargo caused Dresmund's textile output to fall by roughly 60% within two years.
- Paraphrase combining paras 2 and 3.

**P3.** The River Keld blockade lasted three years and caused grain prices to rise eightfold in Dresmund.
- Combines facts from para 4.

**P4.** The Treaty of Veliste granted Cerantia exclusive harvesting rights over Lake Verath's mollusc populations.
- Direct from para 5.

### TRAINING (4) — Real-world facts thematically adjacent

**T1.** In medieval Europe, river blockades were often enforced using chains stretched across the waterway between fortified positions on each bank.
- Real fact about medieval military/economic tactics. The passage describes a river blockade but says nothing about how it was enforced.

**T2.** Freshwater mollusc populations are highly sensitive to changes in water temperature and flow, making them vulnerable to upstream damming or diversion.
- Real ecological fact. The passage describes molluscs in Lake Verath but says nothing about their environmental sensitivities.

**T3.** Medieval guild systems typically required multi-year apprenticeships before a craftsman could operate independently, with durations varying by trade and region.
- Real fact about medieval guilds. The passage mentions merchant guilds declaring insolvency but says nothing about guild structure or training.

**T4.** River tolls were a major source of revenue for medieval European cities situated along navigable waterways, with some rivers having dozens of toll stations along their length.
- Real historical fact. The passage describes cities on the River Keld but says nothing about toll practices or river revenue.

### INFERRED (12) — Range from tight to reasonable

**I1.** Cerantia's economy was likely more diversified than Dresmund's, since Cerantia survived the pellucine embargo better than Dresmund survived without it.
- Passage: Dresmund's textile output fell 60%, guild insolvencies. Cerantia's main suffering was from the later grain blockade, not the embargo. Inference: Cerantia had economic options beyond selling dye.

**I2.** The Accord of Berath in 1298 suggests that Dresmund and Cerantia had a history of trade cooperation spanning nearly fifty years before the conflict.
- Passage: Accord signed 1298, conflict started 1347. Simple arithmetic: 49 years of cooperation.

**I3.** Aldric Venn's overland expeditions to find alternative violet pigments likely failed because the specific freshwater mollusc species was endemic to Lake Verath.
- Passage: molluscs found "only in Lake Verath" + expeditions found nothing comparable. Tight inference.

**I4.** The River Keld blockade was economically irrational for Dresmund since the city also depended on river commerce for food imports.
- Passage states both facts (Dresmund blockaded the river AND depended on it for food). Inference: this was self-harming, i.e., irrational in pure economic terms.

**I5.** The Archon of Veliste likely had economic motivations for mediating, since Veliste was located at the river mouth and would have been affected by disrupted trade.
- Passage: Veliste at mouth of Keld, mediated the treaty. Real-world reasoning: river-mouth cities depend on upstream commerce.

**I6.** The 15-year ban on unilateral tariffs in the Treaty of Veliste was likely intended to prevent a repeat of the 1347 tariff that triggered the wars.
- Passage: 1347 tariff started the wars + treaty banned unilateral tariffs. Straightforward causal inference.

**I7.** Cerantia's exclusive harvesting rights over Lake Verath gave it a de facto monopoly on violet dye production in the Halvarian Basin.
- Passage: Lake Verath was the only source of the molluscs + treaty granted exclusive harvesting rights. Inference: monopoly.

**I8.** The fact that neither city deployed its militia suggests both cities' military forces were roughly comparable, creating mutual deterrence.
- Passage: neither deployed militia, attributed to economic interdependence. Adding real-world reasoning about deterrence. Reasonable but not the only explanation.

**I9.** Dresmund's grain imports from the northern provinces would have been disrupted by its own blockade if those imports also travelled by river.
- Passage: Dresmund depended on river for food from northern provinces + blockaded the river. Inference: self-disruption.

**I10.** The chronicler Marath of Irwen likely wrote after the conflict ended, given the retrospective tone of the quoted passage.
- Passage quotes Marath with a summary judgment. Inference: retrospective accounts are written after events.

**I11.** The joint customs authority's 90-year lifespan suggests the institutional framework of the Treaty of Veliste was more durable than the political entities that created it.
- Passage: authority lasted 90+ years, outlasted political independence of both cities. Inference: institutions outlasted their creators.

**I12.** The development of synthetic dye techniques in the region was an unintended consequence of the trade wars.
- Passage: Dresmund's "failed search for natural alternatives prompted investment in chemical experimentation." The passage says the wars "accelerated" this but doesn't call it unintended. Inference from the narrative structure: the original goal was finding natural alternatives, not inventing synthetic ones.

### FABRICATED (12) — Near-miss items, almost but not quite inferable

**F1.** The Pellucine Council derived its name from the pellucine dye trade, reflecting Cerantia's economic dependence on the pigment industry.
- Why plausible: "Pellucine" Council + pellucine dye = obvious naming connection. But passage never explains the name's origin, and it could equally be a family name, a geographic term, or predate the dye trade.

**F2.** Dresmund's merchant guilds likely lobbied Aldric Venn to impose the 1347 tariff in order to protect domestic wool processing.
- Why plausible: tariffs are typically imposed to protect domestic industry, and merchant guilds would be natural lobbyists. But passage says nothing about who pushed for the tariff or why Dresmund imposed it.

**F3.** The grain shortages caused by the Keld blockade likely led to civil unrest in both cities.
- Why plausible: grain shortages historically cause unrest (real-world knowledge). 8x price increase is severe. But the passage mentions no unrest, riots, or social consequences beyond prices.

**F4.** Cerantia's control of pellucine dye production eventually declined after synthetic alternatives became available.
- Why plausible: the passage mentions synthetic dyes were developed. Logical that a monopoly on natural dye would erode. But the passage says development happened "decades later" and gives no information about Cerantia's subsequent dye trade.

**F5.** The Treaty of Veliste likely included provisions for military non-aggression, given the fifteen years of hostility preceding it.
- Why plausible: peace treaties commonly include such provisions. But the passage lists the treaty's provisions (free trade, joint customs, tariff ban, harvesting rights) and a military clause is not among them. Absence from the list suggests it wasn't included.

**F6.** Marath of Irwen was probably a Cerantian historian, given that Irwen is described as being in the Halvarian Basin.
- Why plausible: Irwen sounds like it could be near Cerantia. But the passage says nothing about Irwen's location or Marath's affiliation — "Irwen" is not described at all.

**F7.** The failure to find alternative violet pigments suggests that Dresmund's expeditions were limited to the immediate Halvarian Basin rather than extending to more distant regions.
- Why plausible: sounds like a reasonable explanation for failure. But the passage says "overland expeditions" without specifying range, and the failure could equally be because the mollusc species was truly unique.

**F8.** The Accord of Berath probably also regulated river navigation rights, given the importance of the River Keld to both cities.
- Why plausible: free-trade agreements between river cities often cover navigation. But the passage only describes the Accord as "a free-trade agreement" and says nothing about its other provisions.

**F9.** The joint customs authority established by the treaty likely employed officials from both Dresmund and Cerantia to ensure impartial enforcement.
- Why plausible: "joint" implies shared governance. But the passage gives no details about the authority's structure or staffing.

**F10.** The Halvarian Trade Wars are sometimes compared by modern historians to the Anglo-Dutch Wars of the 17th century due to their commercial nature.
- Why plausible: sounds like a reasonable historiographic comparison. But the passage makes no such comparison, and this introduces a specific real-world analogy that is purely invented.

**F11.** Veliste's role as mediator enhanced its political status, eventually making it the dominant city-state in the Halvarian Basin.
- Why plausible: mediation often brings prestige. But the passage says the authority "outlasted the political independence of both founding cities" without mentioning Veliste's subsequent trajectory.

**F12.** The 60% decline in Dresmund's textile output would have reduced demand for raw wool, likely harming agricultural communities that supplied the textile industry.
- Why plausible: supply chain reasoning — less textile production = less wool demand = upstream harm. But the passage says nothing about agricultural impacts or wool-producing communities. This is a plausible but unsupported economic extrapolation.

---

## Validation Checklist

- [ ] Every PASSAGE item is verifiable by direct quotation from the text
- [ ] Every TRAINING item is a real-world fact, verifiable independently
- [ ] Every INFERRED item requires combining a passage fact + a real-world fact (inference chain documented)
- [ ] Every FABRICATED item sounds plausible but is NOT stated or derivable
- [ ] No FABRICATED item can be reclassified as INFERRED by a reasonable chain of reasoning
- [ ] Cross-validator (John) can correctly classify all 32 without the answer key
- [ ] Passage contains no real places, people, or events that could leak from training data
