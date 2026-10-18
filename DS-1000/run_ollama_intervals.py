from run_ollama import run_ollama

intervals = [(0,1), (0,1000)]
for i in range(1, len(intervals)):
	run_ollama("granite-code:latest", 0.2, i, True, "", intervals[i][0], intervals[i][1])
