def generator_function():
    received_value = yield "Ready to receive"
    while True:
        result = f"Received: {received_value}"
        received_value = yield result

# Create the generator object
gen = generator_function()

# Start the generator
print(next(gen))  # Output: "Ready to receive"

# Send values to the generator
print(gen.send(10))  # Output: "Received: 10"
print(gen.send(20))  # Output: "Received: 20"
