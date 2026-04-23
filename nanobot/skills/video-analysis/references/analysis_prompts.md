# Video Analysis Prompt Templates

This reference contains prompt templates for different video analysis modes.
Use these as a guide when analyzing frames with Claude Vision API.

---

## Table of Contents

1. [Academic Lecture](#academic-lecture)
2. [Tutorial Video](#tutorial-video)
3. [Demo Video](#demo-video)
4. [Experiment Video](#experiment-video)
5. [Meeting Record](#meeting-record)
6. [General Describe](#general-describe)

---

## Academic Lecture

**Use case**: Academic talks, conference presentations, paper reviews, university lectures

**Prompt template**:
```
Analyze this academic video frame. Extract:
1. Slide content (titles, bullet points, main ideas)
2. Diagrams, charts, or visual illustrations
3. Mathematical equations or formulas
4. Key concepts being discussed
5. Any code snippets or pseudocode
6. Author acknowledgments or references

Be thorough and capture all informational content.
For equations, describe notation and structure.
For diagrams, explain components and relationships.
```

**Best for**:
- Conference paper presentations
- University course lectures
- Technical seminar recordings
- Research methodology discussions

---

## Tutorial Video

**Use case**: Technical tutorials, how-to guides, software demonstrations, training materials

**Prompt template**:
```
Analyze this tutorial video frame. Extract:
1. Current step being demonstrated
2. UI elements and interface being used
3. Any code, commands, or configuration shown
4. Key instructions or tips
5. Expected outcomes or results
6. Prerequisites mentioned (if any)

Focus on actionable information for reproducibility.
Note any keyboard shortcuts, menu paths, or configuration values.
```

**Best for**:
- Software tutorials
- Technical how-to guides
- API documentation videos
- Onboarding/training content

---

## Demo Video

**Use case**: Product demos, feature showcases, system walkthroughs

**Prompt template**:
```
Analyze this demo video frame. Identify:
1. Product or feature being demonstrated
2. User interactions and workflows
3. Key benefits or value propositions shown
4. Interface design elements
5. Notable moments or highlights
6. Any metrics or results displayed

Capture the essence of what makes this demo effective.
Note any unique features or selling points visible.
```

**Best for**:
- Product launch videos
- Feature demonstrations
- SaaS platform tours
- Trade show presentations

---

## Experiment Video

**Use case**: Scientific experiments, lab recordings, process documentation

**Prompt template**:
```
Analyze this experiment video frame. Document:
1. Experimental setup and equipment visible
2. Materials or samples being used
3. Procedures or steps being performed
4. Observable phenomena or results
5. Measurement readings or data displays
6. Safety precautions or protocols

Be precise and objective in your observations.
Note any control elements or variables.
```

**Best for**:
- Lab experiment recordings
- Scientific procedure documentation
- Quality control videos
- Engineering process documentation

---

## Meeting Record

**Use case**: Meeting recordings, conference calls, group discussions

**Prompt template**:
```
Analyze this meeting video frame. Identify:
1. Participants visible (if any)
2. Content on screens or shared displays
3. Key discussion points or topics
4. Decisions or action items mentioned
5. Meeting context and setting
6. Any presentation materials shown

Focus on extracting actionable information.
Note who is speaking if identifiable.
```

**Best for**:
- Business meeting recordings
- Conference call documentation
- Team standup videos
- Webinar recordings

---

## General Describe

**Use case**: General-purpose scene description, content indexing

**Prompt template**:
```
Analyze this video frame. Provide:
1. Scene description - what is happening
2. Key visual elements and objects present
3. Text or UI elements visible
4. Overall context and setting
5. Apparent tone or mood

Be concise and objective.
This description will be used for content indexing.
```

**Best for**:
- Content search indexing
- Video summarization
- Accessibility descriptions
- Archive documentation

---

## Joint Text+Visual Analysis

When transcript context is available, prepend:

```
The audio context around this frame is:
"[transcript segment]"

Based on this context and the visual content:
[analysis prompt]
```

This enables better understanding of:
- What speaker is referring to visually
- Correlation between speech and visuals
- Speaker intent and visual aids

---

## Multi-Frame Analysis

For batch analysis with multiple frames, use:

```
Analyze this set of frames from a video.
The frames are from timestamps: [list]

For each frame, provide:
- Timestamp
- Key visual elements
- Relationship to adjacent frames
- Notable changes or transitions

Finally, summarize the overall narrative or progression.
```

This is useful for:
- Scene understanding
- Action sequence analysis
- Tutorial step tracking

---

## Frame Filtering Criteria

### High-Value Frames (analyze with priority)
- Scene change boundaries
- Text-heavy slides or screens
- Key moments or highlights
- Data visualizations
- Transition frames

### Lower-Value Frames (can be skipped)
- Repeated similar frames
- Blurry or low-quality frames
- Static scenes with no change
- Credits/title screens (unless relevant)

### Quality Thresholds
- Minimum Laplacian variance: 30 (blur detection)
- Maximum similarity: 80% (duplicate detection)
- Minimum text size: 10px (readable text)
