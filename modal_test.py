import modal

app = modal.App("rift-test")

@app.function(cpu=1, timeout=30)
def hello():
    return "ok"

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(result)
