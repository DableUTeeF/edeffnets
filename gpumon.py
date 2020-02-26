import os
import re
import subprocess
import time
import datetime

command = 'nvidia-smi'
max_len = 80
while True:
    os.system('clear')
    t = time.time()
    p = subprocess.check_output(command)

    c = re.findall(r'\b\d+C', str(p))
    temperature = int(c[0][:-1])
    if temperature > 65:
        tempcolor = '\033[33m'
    elif temperature > 75:
        tempcolor = '\033[31m'
    else:
        tempcolor = '\033[32m'
    n = re.findall(r'\b\d+%', str(p))
    fan_percent = int(n[0][:-1])
    gpu_percent = int(n[1][:-1])
    ram_using = re.findall(r'\b\d+MiB+ /', str(p))[0][:-5]
    ram_total = re.findall(r'/  \b\d+MiB', str(p))[0][3:-3]
    ram_percent = int(ram_using) / int(ram_total)
    if 0 < int(ram_using) < 4000:
        ram_color = '\033[32m'
    elif int(ram_using) < 6500:
        ram_color = '\033[33m'
    else:
        ram_color = '\033[31m'
    ram1 = '\033[0mMemory-Usage['
    ram2 = f'{ram_color}{"|"*int(ram_percent/2*100)}{" "*(50-int(ram_percent/2*100))}\033[0m]:{ram_using}/{ram_total}'
    ram = ram1+ram2
    a = len(ram)
    buffer = " "*(93-len(ram))
    if gpu_percent < 50:
        gpu_color = '\033[32m'
    elif gpu_percent < 90:
        gpu_color = '\033[33m'
    else:
        gpu_color = '\033[31m'
    if fan_percent < 50:
        fan_color = '\033[32m'
    elif fan_percent < 75:
        fan_color = '\033[33m'
    else:
        fan_color = '\033[31m'
    util1 = '\033[0mGPU-Util    ['
    util2 = f'{gpu_color}{"|"*int(gpu_percent/2)}{" "*(50-int(gpu_percent/2))}\033[0m]: {gpu_percent}%    '
    util = util1+util2
    print('GeForce GTX 1080')
    reset_color = '\033[0m'
    print(f'Fan: {fan_color}{n[0]}{reset_color}, Temp: {tempcolor}{c[0]}')
    print(ram+buffer+util, end='\r')
    if temperature > 84:
        print(str(datetime.datetime.now()), 'temperature is', temperature, 'celcius')
        print('Suicide Bombing')
        print(subprocess.check_output('killall python'))
    time.sleep(1-(time.time()-t))
