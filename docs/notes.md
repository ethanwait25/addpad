# Rewards
- +1.0 - correct answer
- -1.0 - incorrect answer
- -0.1 - illegal action
- -0.01 - per step

# Timeline
1. Spec out the "game"
2. Build the gym-style environment
3. Make oracle solver to produce exact action sequence
4. Create and train small behavior cloner to imitate the oracle
5. Create RL model - to be trained A) from scratch and B) using BC initialization of weights
6. Implement curriculum - 0-9 no carry, 0-9 carry, 2-digit no carry, 2-digit carry, 3-digit
7. Attempt "finger counting learning"
8. Compare oracle, BC, from-scratch RL, BC-initialized RL, and "finger counting" RL