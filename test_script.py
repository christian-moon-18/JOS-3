# Step 0: Import required packages
import jos3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

height_inch=70
weight_lb=160

# Step 1: Initialize the JOS3 model with body characteristics
model = jos3.JOS3(
    height=height_inch/39.3701,             # Body height in meters
    weight=weight_lb/2.20462,            # Body weight in kg
    fat=15,                  # Body fat percentage
    age=25,                  # Age in years
    sex="male",              # "male" or "female"
    ci=2.6432,               # Cardiac index [L/min/m2]
    bmr_equation="harris-benedict",  # BMR equation: "harris-benedict" or "japanese"
    bsa_equation="dubois",   # BSA equation: "dubois", "fujimoto", "kruazumi", "takahira"
    ex_output="all"          # Output all parameters; use list like ["BFsk", "Tsk"] for specific ones
)

# Step 2: Set initial environmental conditions
model.Ta = 28               # Air temperature [°C]
model.Tr = 30               # Mean radiant temperature [°C]
model.RH = 40               # Relative humidity [%]
model.Va = 0.2              # Air velocity [m/s]
model.PAR = 1.2             # Physical activity ratio (1.2 = sitting quietly)
model.posture = "lying"   # Posture: "standing", "sitting", or "lying"

# Clothing insulation per body part (17 segments)
model.Icl = np.array([
    0.00,  # Head
    0.00,  # Neck
    1.14,  # Chest
    0.84,  # Back
    1.04,  # Pelvis
    0.84,  # Left-Shoulder
    0.42,  # Left-Arm
    0.00,  # Left-Hand
    0.84,  # Right-Shoulder
    0.42,  # Right-Arm
    0.00,  # Right-Hand
    0.58,  # Left-Thigh
    0.62,  # Left-Leg
    0.82,  # Left-Foot
    0.58,  # Right-Thigh
    0.62,  # Right-Leg
    0.82   # Right-Foot
])

# Step 3: Run simulation for initial condition
model.simulate(
    times=30,   # Number of simulation loops
    dtime=60    # Time step in seconds (default is 60)
)

# Step 4: Change environment for transient simulation
model.To = 20  # Change operative temperature
model.Va = np.array([  # Change air velocity per body part
    0.2, 0.4, 0.4, 0.1, 0.1, 0.4, 0.4, 0.4,
    0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
])

model.simulate(times=60, dtime=60)

# Step 5: Further change environment
model.Ta = 30
model.Tr = 35

model.simulate(times=30, dtime=60)


# Step 6: Export and visualize results with labels
df = pd.DataFrame(model.dict_results())

plt.figure(figsize=(10, 6))
plt.plot(df["Tcb"], label="Central blood pool temperature")
plt.xlabel("Time Step")  # Each step = dtime seconds
plt.ylabel("Temperature (°C)")
plt.title("Mean Core Blood Pool Temperature Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("mean_cbp_temperature_plot.png")  # Save figure
plt.show()  # Display plot

# Save results to CSV
model.to_csv("example.csv")

# Optional: Print documentation of output parameters
print(jos3.show_outparam_docs())

# Optional: Use getters to inspect internal model state
print("Basal Metabolic Rate:", model.BMR)
print("Mean Skin Temperature:", model.TskMean)
print("Skin Temperatures:", model.Tsk)