# Passage 003: Velnic Notation

## Design Notes
- **Domain:** Formal/technical (mathematical notation system)
- **Why this domain:** Tests logical/structural reasoning with minimal reliance on world knowledge. The rules are self-contained in the passage, so INFERRED items require applying those rules rather than importing external facts. This is the passage where TRAINING items are hardest to write (few real-world facts connect to a fictional notation system) but INFERRED items are most unambiguous (logical derivations from stated rules).
- **Contamination risk:** Zero. "Velnic notation" is entirely invented.

---

## Passage (478 words)

Velnic notation is a positional writing system for representing rational numbers, developed in 2021 by the mathematician Elen Barudi as an alternative to standard decimal and fractional representations. The system uses base-6 encoding and represents every rational number as a finite string of symbols drawn from the set {0, 1, 2, 3, 4, 5}, augmented by a single separator mark called the "pivot," written as a vertical bar (|).

In Velnic notation, the digits to the left of the pivot represent the integer part of the number in standard base-6, read left to right from the most significant digit. The digits to the right of the pivot encode the fractional part using a mirrored place-value system: the first digit after the pivot represents sixths, the second represents thirty-sixths, and so on, with each position worth one-sixth of the preceding position.

A key design choice in Velnic notation is the absence of a negative sign. Negative numbers are instead represented by complementing every digit in the string — replacing each digit d with (5 - d) — and appending the symbol "c" to the end of the representation. For example, if the Velnic representation of 7 is 11|0, then -7 is written as 44|5c, since each digit (1, 1, 0) is replaced by (4, 4, 5).

Barudi introduced Velnic notation in a self-published monograph titled "Positional Alternatives," which was initially circulated only within a small online community of recreational mathematicians. The system gained wider attention in 2023 when it was featured in a competition problem at the Baltic Mathematical Olympiad, where contestants were asked to perform arithmetic operations in Velnic encoding. The problem was subsequently discussed in several mathematics education blogs, though no university curriculum has adopted Velnic notation for instruction.

One notable property of Velnic notation is that all rational numbers with denominators that are products of 2 and 3 (and no other prime factors) have terminating representations. This is because 6 = 2 × 3, and the base determines which fractions terminate. For instance, 1/4 has a terminating Velnic representation (since 4 = 2², and 2 divides 6), while 1/7 does not terminate and must be written as a repeating sequence.

Barudi has also proposed an extension called "augmented Velnic" that introduces two additional symbols (A and B, representing values 6 and 7 in each position) to create a base-8 variant. Augmented Velnic would allow terminating representations of fractions with denominators that are products of 2 only, since 8 = 2³. However, this extension has not been formally published and exists only as a draft shared on Barudi's personal website.

The Velnic system has been praised by some recreational mathematicians for its elegance in handling the complement-based negation scheme, but critics have noted that the base-6 encoding makes mental arithmetic more difficult for users accustomed to base-10, and that the practical advantages over existing notation systems remain unclear.

---

## Statements (32 total)

### PASSAGE (4)

**P1.** Velnic notation uses base-6 encoding with the digit set {0, 1, 2, 3, 4, 5}.
- Sentence: "The system uses base-6 encoding" + "string of symbols drawn from the set {0, 1, 2, 3, 4, 5}."

**P2.** Negative numbers are represented by complementing every digit and appending "c".
- Sentence: "Negative numbers are instead represented by complementing every digit in the string — replacing each digit d with (5 - d) — and appending the symbol 'c'."

**P3.** Velnic notation was featured in a 2023 Baltic Mathematical Olympiad competition problem.
- Sentence: "gained wider attention in 2023 when it was featured in a competition problem at the Baltic Mathematical Olympiad."

**P4.** Augmented Velnic introduces symbols A and B to create a base-8 variant.
- Sentence: "an extension called 'augmented Velnic' that introduces two additional symbols (A and B, representing values 6 and 7) to create a base-8 variant."

### TRAINING (4)

**T1.** In base-10, the fraction 1/3 has a non-terminating repeating decimal representation (0.333...).
- Real maths fact. Passage discusses termination but only in base-6 context.

**T2.** The Baltic Mathematical Olympiad is an annual competition for secondary school students from countries in the Baltic region.
- Real fact about a real competition named in the passage. Passage doesn't describe the competition.

**T3.** A rational number is any number that can be expressed as the ratio of two integers.
- Real maths definition. Passage uses "rational numbers" but doesn't define the term.

**T4.** In standard positional notation, base-16 (hexadecimal) uses the symbols 0-9 and A-F.
- Real fact about another base system. Passage discusses base-6 and base-8 but not base-16.

### INFERRED (12)

**Tight (4):**

**I1.** In Velnic notation, the number 0 is represented as 0|0.
- Passage: integer part left of pivot in base-6, fractional part right of pivot. World: 0 in any base is 0. Therefore: 0|0.

**I2.** The Velnic representation of -7 (44|5c) has the same number of digits as the representation of 7 (11|0), plus the "c" suffix.
- Passage: gives both representations explicitly (11|0 and 44|5c). Logic: counting characters. 3 digits each + c.

**I3.** In Velnic notation, the fraction 1/6 is represented as 0|1.
- Passage: first digit after pivot represents sixths. World: 1/6 = one sixth. Therefore: 0 integer part, 1 in the sixths place = 0|1.

**I4.** The complement of 0|0 in Velnic notation is 5|5c, which represents -0.
- Passage: complement replaces each d with (5-d) and appends c. Logic: complement each digit of 0|0 → 5|5, append c → 5|5c. This represents -0.

**Moderate (4):**

**I5.** The fraction 1/2 has a terminating representation in Velnic notation because 2 is a factor of 6.
- Passage: fractions with denominators that are products of 2 and 3 terminate, because 6 = 2 × 3. World: 1/2 has denominator 2, which is a factor of 6. Therefore: terminates.

**I6.** The fraction 1/5 does not have a terminating representation in Velnic notation.
- Passage: terminating iff denominator is a product of 2 and 3 only. World: 5 is neither 2 nor 3. Therefore: does not terminate.

**I7.** In augmented Velnic (base-8), the fraction 1/3 would NOT terminate, since 3 does not divide 8.
- Passage: augmented Velnic is base-8, termination depends on the base. World: 8 = 2³, so only powers of 2 terminate. 3 does not divide 8. Therefore: 1/3 does not terminate.

**I8.** The complement negation scheme means that addition of a number and its complement always yields a string of 5s followed by "c", plus a carry.
- Passage: complement replaces d with 5-d. World: d + (5-d) = 5 for each position. Therefore: adding a number and its complement gives all 5s in each position (plus carry handling at the boundary).

**Hard (4):**

**I9.** Velnic notation could represent any fraction expressible in base-10, since both systems can represent all rational numbers (just with different termination properties).
- Passage: represents rational numbers. World: rational numbers are the same regardless of base; base only affects termination. Therefore: any base-10 fraction is representable.

**I10.** The Baltic Olympiad problem likely required contestants to convert between base-10 and base-6 representations.
- Passage: contestants performed arithmetic in Velnic encoding. World: contestants accustomed to base-10 would need to convert to work in base-6. Therefore: conversion likely required.

**I11.** If Velnic notation were extended to base-12, fractions with denominators that are products of 2 and 3 would still terminate, but so would fractions with denominator 4 (since 4 divides 12).
- Passage: termination depends on the base's prime factors. World: 12 = 2² × 3. 4 = 2², which divides 12. Therefore: 1/4 terminates in base-12.

**I12.** The absence of a negative sign makes Velnic notation's character set fixed-width (each symbol is drawn from the same finite set), which could simplify parsing by computer programs.
- Passage: no negative sign, complement + "c" suffix instead. World: fixed character sets simplify parsing. Therefore: parsing advantage.

### FABRICATED (12)

**Obvious (4):**

**F1.** Elen Barudi is a professor of mathematics at the University of Tartu in Estonia.
- Gap: passage says "mathematician" and "self-published monograph." No affiliation mentioned.

**F2.** Velnic notation has been implemented in the Python programming language as an open-source library.
- Gap: passage mentions only blog discussions. No software implementations mentioned.

**F3.** The monograph "Positional Alternatives" was published by Cambridge University Press in 2022.
- Gap: passage explicitly says "self-published." Contradicts the passage.

**F4.** The International Mathematical Union has recognised Velnic notation as an official alternative to decimal representation.
- Gap: passage says no university curriculum has adopted it. Recognition by IMU not mentioned.

**Near-miss (4):**

**F5.** The complement negation scheme in Velnic notation is analogous to two's complement representation used in binary computing.
- Gap: while there IS a structural analogy (complement-based negation), the passage does not make this comparison. The analogy requires knowledge of two's complement AND a judgment that the structures are analogous. The passage's complement scheme (5-d for each digit) is actually closer to nine's complement in decimal or one's complement in binary, not two's complement (which adds 1 after complementing). So the analogy is imprecise, not just unstated.

**F6.** The base-6 encoding was chosen because 6 is the smallest number with more than two prime factors that include both 2 and 3.
- Gap: 6 = 2 × 3, which is two prime factors, not "more than two." And the passage never explains WHY base-6 was chosen.

**F7.** Augmented Velnic's base-8 system would be more efficient than standard Velnic for representing fractions arising in computer science, since computer architectures are binary.
- Gap: while 8 = 2³ is related to binary, "more efficient" is not derivable. Efficiency depends on the use case, and "fractions arising in computer science" is vague. The passage says augmented Velnic allows terminating representations for powers-of-2 denominators but makes no efficiency claims.

**F8.** The repeating representation of 1/7 in Velnic notation has a period of 6 digits.
- Gap: passage says 1/7 doesn't terminate but gives no detail about the period. The period of 1/7 in base-6 is actually 6 (since the multiplicative order of 6 mod 7 is 6), but this requires calculation not presented in the passage, and the result depends on number theory not stated in the passage. The claim sounds precise and correct but its derivation is not available from the passage.

**Adversarial (4):**

**F9.** Since 1/4 terminates in Velnic notation and 1/4 also terminates in decimal, Velnic notation terminates for exactly the same fractions as decimal notation.
- Gap: the passage says 1/4 terminates in Velnic because 4 = 2² and 2 divides 6. But decimal terminates when the denominator is a product of 2 and 5 (not 2 and 3). So 1/3 terminates in Velnic but not in decimal, and 1/5 terminates in decimal but not in Velnic. The premise (1/4 terminates in both) is true, but the conclusion (same set of terminating fractions) is false.

**F10.** Velnic notation's complement scheme guarantees that subtraction can be performed using addition alone, eliminating the need for a separate subtraction algorithm.
- Gap: this is true of complement arithmetic systems in general (real CS knowledge), and the passage describes a complement scheme, but the passage never claims this property. The model would need to infer that complement-based negation implies complement-based subtraction, which requires assuming the arithmetic operations work the same way as in other complement systems -- an assumption not established for Velnic.

**F11.** The fact that no university has adopted Velnic notation suggests that it offers no practical advantages over existing systems.
- Gap: absence of adoption does not prove absence of advantages. The passage notes critics have questioned practical advantages but also notes praise from recreational mathematicians. "No adoption therefore no advantages" is a logical fallacy (argument from absence).

**F12.** Barudi's choice to self-publish indicates that the monograph was rejected by peer-reviewed journals.
- Gap: self-publishing has many possible motivations. Rejection is one, but so is speed, audience targeting, or personal preference. The passage gives no reason for the choice.

---

## Validation Checklist

- [ ] PASSAGE: all 4 items have quoted sentences
- [ ] TRAINING: all 4 items are verifiable real-world facts with zero passage dependency
- [ ] INFERRED: all 12 items have complete chains ("Passage says X. World says Y. Therefore Z.")
- [ ] FABRICATED: all 12 items have documented gaps
- [ ] Difficulty tiers assigned: tight/moderate/hard for INFERRED, obvious/near-miss/adversarial for FABRICATED
- [ ] No item requires >2 sentences to justify
- [ ] Cross-validation: second person blind-classifies 8+ items with <15% disagreement
