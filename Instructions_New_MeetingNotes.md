### Project Overview

- January status check and team coordination for digital twins project
- Meeting occurred with Georg joining mid-conversation, some participants (Stefano) were unable to attend despite invitation
- Discussion of upcoming team meeting invitation from external stakeholders regarding digital twins and predictive modeling work
- Decision to postpone external presentation until internal materials and models are better organized
- Focus on clarifying data sources, model architecture, and preparing onboarding materials for new working student

### The What

- Patient exchange modeling system with multiple interconnected models
  - Sequence prediction model for examination workflows
  - Patient exchange model handling transitions between body regions
  - Temporal model managing duration and timestamps
  - Patient conditioning model (currently ahead of schedule and not fully functional)
- Daily workflow simulation capability
  - Generate complete day of synthetic examination data
  - Model patient arrivals and examination sequences
  - Predict transitions between different body regions (knee to head, etc.)
- Dashboard integration for metrics comparison
  - Real-world vs predicted data comparison
  - Customer-specific model performance tracking
  - Flexible data exploration and drill-down capabilities

### Methodology

- Customer-specific modeling approach initially, with future plans for cross-customer models
- Sequential alternating model execution
  1. Exchange model called first (start to next body region)
  2. Examination model processes specific body region
  3. Iterative process continues through patient sequence
  4. Final exchange model handles transition to end state
- Token-based sequence generation for patient exchanges
- Ground truth patient data used for initial training phases
- Bucket-based validation approach
  - Generate multiple samples (up to 1,000) per body region transition
  - Enable random sampling for day simulation without re-running models
- Data parameters include: patient weight, age, height, direction, body regions (from/to), examination duration

### Decisions & Blockers

- **Agreed**: Postpone external team meeting presentation for several weeks
- **Agreed**: Focus on customer-specific models before expanding to multi-customer approach
- **Agreed**: Use ground truth patient sequences initially, defer synthetic patient scheduling modeling
- **Blocker**: Mismatch between sequence data and patient exchange data availability
- **Blocker**: Need clear definition of input/output parameters for dashboard integration
- **Blocker**: Model architecture in flux due to new data integration requirements
- **Deferred**: Patient conditioning implementation (too advanced for current phase)
- **Deferred**: Complete synthetic day generation (modeling patient arrival patterns)

### Next Steps

- Luke to complete 10-slide presentation by Monday for team review
- Team review session to cover presentation content and onboarding materials for new working student
- Define clear parameters for input/output data columns for dashboard integration
- Clarify which data extracts from event logs will be used in current models
- Organize existing information into coherent PowerPoint presentations
- Prepare for eventual external stakeholder presentation once internal materials are ready

