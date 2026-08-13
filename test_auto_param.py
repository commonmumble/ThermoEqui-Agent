"""Quick test for auto_lookup_parameters using production YAML files."""
from agent.orchestrator import ConversationOrchestrator
from schemas.domain import TaskManifest, ComponentIdentity, ThermodynamicConditions

# Test 1: Wilson ethanol-water (should find)
task = TaskManifest(
    calculation_type="bubble_point",
    equilibrium_type="VLE",
    model_name="Wilson",
    components=[
        ComponentIdentity(name="Ethanol", cas_number="64-17-5", component_id="ethanol"),
        ComponentIdentity(name="Water", cas_number="7732-18-5", component_id="water"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.3, 0.7]),
    parameters=[],
)
result = ConversationOrchestrator._auto_lookup_parameters(task)
if result:
    ps = result[0]
    print("PASS: Wilson parameters found for ethanol-water")
    print(f"  ParameterSet ID: {ps.parameter_set_id}")
    print(f"  Parameters: {ps.parameters}")
    print(f"  Form: {ps.parameter_form}")
else:
    print("FAIL: No Wilson parameters found for ethanol-water")

# Test 2: NRTL ethanol-water (should find)
task2 = task.model_copy(update={"model_name": "NRTL", "parameters": []})
result2 = ConversationOrchestrator._auto_lookup_parameters(task2)
if result2:
    ps2 = result2[0]
    print("PASS: NRTL parameters found for ethanol-water")
    print(f"  ParameterSet ID: {ps2.parameter_set_id}")
    print(f"  Parameters: {ps2.parameters}")
else:
    print("FAIL: No NRTL parameters found for ethanol-water")

# Test 3: Reverse order water-ethanol (should find)
task_rev = TaskManifest(
    calculation_type="bubble_point",
    equilibrium_type="VLE",
    model_name="Wilson",
    components=[
        ComponentIdentity(name="Water", cas_number="7732-18-5", component_id="water"),
        ComponentIdentity(name="Ethanol", cas_number="64-17-5", component_id="ethanol"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.3, 0.7]),
    parameters=[],
)
result_rev = ConversationOrchestrator._auto_lookup_parameters(task_rev)
if result_rev:
    print("PASS: Wilson parameters found for water-ethanol (reverse order)")
else:
    print("FAIL: No Wilson parameters found for water-ethanol (reverse order)")

# Test 4: Non-existent system (methane-ethane in Wilson, should NOT find)
task3 = TaskManifest(
    calculation_type="bubble_point",
    equilibrium_type="VLE",
    model_name="Wilson",
    components=[
        ComponentIdentity(name="Methane", cas_number="74-82-8", component_id="methane"),
        ComponentIdentity(name="Ethane", cas_number="74-84-0", component_id="ethane"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.3, 0.7]),
    parameters=[],
)
result3 = ConversationOrchestrator._auto_lookup_parameters(task3)
if not result3:
    print("PASS: No Wilson parameters for methane-ethane (expected)")
else:
    print("FAIL: Unexpectedly found Wilson parameters for methane-ethane")

# Test 5: Task with existing parameters (should NOT override)
task4 = task.model_copy(update={"parameters": result})
result4 = ConversationOrchestrator._auto_lookup_parameters(task4)
if not result4:
    print("PASS: Task with existing params not overridden")
else:
    print("FAIL: Task with existing params was overridden")

# Test 6: UNIQUAC ethanol-water (should find)
task5 = task.model_copy(update={"model_name": "UNIQUAC", "parameters": []})
result5 = ConversationOrchestrator._auto_lookup_parameters(task5)
if result5:
    print("PASS: UNIQUAC parameters found for ethanol-water")
    print(f"  ParameterSet ID: {result5[0].parameter_set_id}")
else:
    print("FAIL: No UNIQUAC parameters found for ethanol-water")

# Test 7: methanol-water NRTL (should find - exists in YAML)
task6 = TaskManifest(
    calculation_type="bubble_point",
    equilibrium_type="VLE",
    model_name="NRTL",
    components=[
        ComponentIdentity(name="Methanol", cas_number="67-56-1", component_id="methanol"),
        ComponentIdentity(name="Water", cas_number="7732-18-5", component_id="water"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.3, 0.7]),
    parameters=[],
)
result6 = ConversationOrchestrator._auto_lookup_parameters(task6)
if result6:
    print("PASS: NRTL parameters found for methanol-water")
    print(f"  ParameterSet ID: {result6[0].parameter_set_id}")
else:
    print("FAIL: No NRTL parameters found for methanol-water")

print("\n=== All tests completed ===")