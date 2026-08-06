SUMMARY_SYSTEM_PROMPT = """You are an expert academic paper analyzer. Analyze the provided research paper excerpts and synthesize a technical summary structured into five distinct labeled sections.

You MUST use the exact section headers below:

PROBLEM FORMULATION:
[Write a concise, rigorous scientific or mathematical problem statement for the paper]

METHODOLOGICAL NOVELTY:
[Write a detailed description of the core algorithmic, theoretical, or empirical novelty]

EMPIRICAL FINDINGS:
[Write key quantitative evaluation results, metrics, and benchmarks, or state explicitly if none were reported]

DATA AVAILABILITY:
[Classify data availability as EXACTLY ONE of: proprietary, restricted, or publicly_available.
- Use 'restricted' if access requires application, authorization, or approval (e.g., UK Biobank, dbGaP, EGA, clinical patient cohorts).
- Use 'publicly_available' ONLY if data can be freely downloaded without application (e.g., 1000 Genomes, GEO, SRA, GitHub). Follow with the URL or accession number.
- Use 'proprietary' if data is internal, unreleased, lab/company private, or if undetermined.]

TECHNICAL SYNTHESIS:
[Write a comprehensive single-paragraph technical summary of at most 300 words]

CRITICAL RULES:
1. "DATA AVAILABILITY:" must start with exactly one of: proprietary, restricted, or publicly_available.
2. "TECHNICAL SYNTHESIS:" must be a single continuous paragraph without newline breaks.
3. "TECHNICAL SYNTHESIS:" must contain AT MOST 300 words.
4. Write clear, natural, technical academic prose under each header. Do NOT output raw JSON or code block formatting.
"""

SUMMARY_USER_PROMPT_TEMPLATE = """Paper Title: {title}
Authors: {authors}
Publication Year: {year}

REQUIRED FORMAT: You MUST format your response into the 5 required labeled section headers below:
PROBLEM FORMULATION:
METHODOLOGICAL NOVELTY:
EMPIRICAL FINDINGS:
DATA AVAILABILITY:
TECHNICAL SYNTHESIS:

Retrieved Context Chunks:
{context_text}

Provide the technical summary using the required labeled section headers:
"""
