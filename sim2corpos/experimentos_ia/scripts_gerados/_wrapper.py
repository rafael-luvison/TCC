import sys, runpy, resource, json
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
finally:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak = ru.ru_maxrss/1e6 if sys.platform=="darwin" else ru.ru_maxrss/1024.0
    sys.stderr.write("\n__METRICS__ "+json.dumps({"peak_rss_mb":peak,"cpu_s":ru.ru_utime+ru.ru_stime})+"\n")
