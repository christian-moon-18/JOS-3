# CLAUDE.md - JOS3 Heat Transfer Visualization Development Guide

**Project:** JOS3 Heat Transfer Visualization Module  
**Version:** 1.0  
**Last Updated:** January 2025  
**Purpose:** AI-Assisted Development Guide

---

## 📖 Required Reading

**Before beginning any development work, you MUST read and understand these foundational documents:**

1. **Product Requirements Document (PRD)** - `JOS3 Heat Transfer Visualization - Product Requirements Document.md`
   - Contains complete functional requirements, user stories, and success metrics
   - Defines the vision, target users, and key benefits
   - Specifies all visualization modes and technical architecture options

2. **Technical Design Document (TDD)** - `JOS3 Heat Transfer Visualization - Technical Design Document.md`
   - Details system architecture, module structure, and implementation approach
   - Contains core algorithms, especially external heat calculation methodology
   - Specifies technology stack, performance requirements, and data processing pipeline

3. **Agile Project Plan** - `JOS3 Heat Transfer Visualization - Agile Project Plan.md`
   - Defines 12-week development timeline with 6 sprints
   - Contains detailed user stories, tasks, and acceptance criteria
   - Specifies sprint goals, definitions of done, and risk management

**⚠️ CRITICAL:** Always reference these documents when implementing features or making design decisions. Do not duplicate information that exists in these files.

---

## 🤖 AI Development Instructions

### Core Development Approach

**Context Loading Protocol:**
1. Always read the relevant sections from PRD/TDD/Project Plan before coding
2. Reference specific requirements, user stories, or technical specifications
3. Validate implementation against documented acceptance criteria
4. Cite document sections when explaining design decisions

**Implementation Priority:**
- Follow the sprint order defined in the Agile Project Plan
- Implement exact functionality specified in PRD functional requirements
- Use technical architecture and algorithms from TDD
- Meet all performance targets and quality standards

### Code Generation Guidelines

**When implementing any feature:**
```python
# REQUIRED: Reference the source requirement
"""
Implementation of [Feature Name]
Source: [PRD/TDD/Plan] Section [X.Y.Z]
User Story: [Exact story from project plan]
Acceptance Criteria: [List from project documents]
"""
```

**Key Implementation Notes:**
- Use exact class names, method signatures, and module structure from TDD Section 2.2
- Implement heat calculation algorithm exactly as specified in TDD Section 5
- Follow performance requirements from PRD Section 5.2
- Use technology stack recommendations from TDD Section 4.1

### Sprint Execution Protocol

**For each sprint, reference the Agile Project Plan:**
1. Read sprint goals and user stories
2. Review specific tasks and acceptance criteria
3. Implement according to technical specifications in TDD
4. Validate against functional requirements in PRD
5. Ensure definition of done criteria are met

**Example Sprint Implementation Flow:**
```
1. Read: Agile Plan → Sprint 1 → Epic 1.2 → Tasks 1.2.1-1.2.3
2. Reference: TDD → Section 3.1 → Data Parser Module
3. Validate: PRD → Section 3.2 → Data Integration requirements
4. Implement with full context from all three documents
```

---

## 🔧 Development Environment Setup

### Prerequisites Verification
```bash
# Verify you have access to all project documents
ls project-resources/
# Should show:
# - JOS3 Heat Transfer Visualization - Product Requirements Document.md
# - JOS3 Heat Transfer Visualization - Technical Design Document.md  
# - JOS3 Heat Transfer Visualization - Agile Project Plan.md
```

### Project Structure
Use the exact structure defined in **TDD Section 2.2** - do not modify without updating the technical design document.

### Dependencies
Install packages as specified in **TDD Section 4.1** and **Agile Plan requirements.txt**.

---

## 📋 Current Sprint Focus

**Sprint Status:** [To be updated as development progresses]

**Active Sprint:** [Current sprint from Agile Project Plan]

**Current Epic/User Stories:** [Reference specific stories from plan]

**Implementation Tasks:** [Reference current sprint tasks]

**Acceptance Criteria:** [Copy from project plan - do not duplicate here]

---

## ✅ Quality Assurance Checklist

**Before considering any feature complete:**

- [ ] Functionality matches PRD requirements exactly
- [ ] Implementation follows TDD architecture specifications  
- [ ] All acceptance criteria from Agile Plan are met
- [ ] Performance targets from PRD Section 5.2 are validated
- [ ] Code structure matches TDD module organization
- [ ] Testing requirements from Agile Plan Definition of Done are satisfied

---

## 🐛 Problem Resolution Protocol

**When encountering issues:**

1. **First:** Check if the problem is addressed in PRD assumptions or constraints
2. **Second:** Review TDD architecture for technical solutions
3. **Third:** Reference Agile Plan risk mitigation strategies
4. **Fourth:** Update this CLAUDE.md with resolution for future reference

**Common Reference Points:**
- Algorithm issues → TDD Section 5 (External Heat Calculation)
- Performance problems → PRD Section 5.2 + Agile Plan Phase 2 optimization tasks
- User interface questions → PRD Section 3.1 (Core Visualization Features)
- Architecture decisions → TDD Section 2 (System Architecture)

---

## 📝 Documentation Standards

**Code Documentation:**
- Reference PRD/TDD section numbers in docstrings
- Link user stories to implementation
- Cite specific requirements being fulfilled

**Example:**
```python
def calculate_external_heat_segment(segment_data, segment_name, time_index):
    """
    Calculate external heating/cooling power for body segment.
    
    Implements: TDD Section 5.2 - External Heat Calculation Algorithm
    Fulfills: PRD Section 3.2.2 - Calculated Metrics requirement
    User Story: From Agile Plan Sprint 1, Epic 1.3, Task 1.3.1
    
    Algorithm follows exact specification in TDD Section 5.3.
    """
```

---

## 🚀 AI Prompt Templates

**For Feature Implementation:**
```
I need to implement [feature name] for the JOS3 Heat Transfer Visualization project.

Requirements Source: [PRD Section X.Y] 
Technical Spec: [TDD Section X.Y]
User Story: [From Agile Plan Sprint X]

Please implement according to the specifications in the referenced documents, ensuring all acceptance criteria are met.
```

**For Bug Fixes:**
```
I'm encountering [issue description] in the JOS3 visualization project.

Context: [Current sprint and task from Agile Plan]
Expected Behavior: [From PRD requirements]
Technical Implementation: [Reference TDD section]

Please provide a solution that aligns with the documented architecture and requirements.
```

**For Performance Optimization:**
```
I need to optimize [component] to meet performance requirements.

Target: [Specific requirement from PRD Section 5.2]
Current Implementation: [Reference TDD section]
Optimization Context: [Sprint task from Agile Plan]

Please suggest improvements that maintain the documented architecture.
```

---

## 📊 Progress Tracking

**Sprint Completion Status:**
- Sprint 1: [ ] Not Started / [ ] In Progress / [ ] Complete
- Sprint 2: [ ] Not Started / [ ] In Progress / [ ] Complete
- Sprint 3: [ ] Not Started / [ ] In Progress / [ ] Complete
- Sprint 4: [ ] Not Started / [ ] In Progress / [ ] Complete
- Sprint 5: [ ] Not Started / [ ] In Progress / [ ] Complete
- Sprint 6: [ ] Not Started / [ ] In Progress / [ ] Complete

**Key Milestones:**
- Phase 1 Complete (Week 4): [ ] 2D visualization and external heat calculation
- Phase 2 Complete (Week 8): [ ] Full 3D visualization with video export  
- Phase 3 Complete (Week 12): [ ] Production-ready package with documentation

---

## 🔄 Document Updates

**This CLAUDE.md should be updated when:**
- Sprint status changes
- New implementation insights are discovered
- Architecture decisions require clarification
- Development bottlenecks are resolved
- Performance optimizations are implemented

**Update Protocol:**
1. Reference the change back to source documents (PRD/TDD/Plan)
2. If source documents need updates, update those first
3. Keep this guide focused on AI development process, not requirements

---

## 🎯 Success Metrics

**Primary Success Indicators:**
- All functional requirements from PRD are implemented
- Technical architecture from TDD is followed
- Sprint goals from Agile Plan are met on schedule
- Final deliverable meets all acceptance criteria

**Quality Gates:**
- Code matches documented architecture (TDD)
- Features fulfill documented requirements (PRD)  
- Implementation timeline follows project plan
- Performance meets specified targets

---

## 📞 Final Notes

This guide serves as the AI development coordinator for the JOS3 Heat Transfer Visualization project. It does NOT replace the comprehensive requirements, technical specifications, or project planning found in the core documents.

**Remember:** The authoritative source for WHAT to build is the PRD, HOW to build it is the TDD, and WHEN to build it is the Agile Project Plan. This CLAUDE.md simply orchestrates AI-assisted development across these documents.

**Success depends on:** Faithful implementation of documented requirements using documented architecture following the documented timeline.