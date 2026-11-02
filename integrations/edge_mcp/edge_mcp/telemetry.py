"""Read-only remote snapshot; sent over SSH stdin, never installed."""
import csv, glob, json, os, shutil, subprocess, sys, time
errors = []
def command(args, timeout=6):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if p.returncode:
            errors.append(args[0] + ': ' + p.stderr.strip()[:350]); return ''
        return p.stdout
    except Exception as e:
        errors.append(args[0] + ': ' + str(e)); return ''
def cpu():
    result = {}
    for line in open('/proc/stat'):
        parts = line.split()
        if not parts or not parts[0].startswith('cpu'): continue
        v = list(map(int, parts[1:9]))
        result[parts[0]] = (sum(v), v[3] + v[4])
    return result

def fast_snapshot():
    a = cpu(); time.sleep(.5); b = cpu()
    def usage(key):
        before, after = a[key], b[key]
        return round(max(0,min(100,100*(1-(after[1]-before[1])/max(1,after[0]-before[0])))),1)
    mem = {l.split(':')[0]: int(l.split()[1]) * 1024 for l in open('/proc/meminfo')}
    net = {}
    for line in open('/proc/net/dev').readlines()[2:]:
        name, values = line.split(':'); v = values.split()
        if name.strip() != 'lo': net[name.strip()] = {'rx':int(v[0]), 'tx':int(v[8])}
    temps = []
    for path in glob.glob('/sys/class/hwmon/hwmon*/temp*_input'):
        try:
            value = float(open(path).read()) / 1000
            stem = path[:-6]; label = stem + '_label'
            name = open(os.path.dirname(path)+'/name').read().strip()
            label = open(label).read().strip() if os.path.exists(label) else os.path.basename(stem)
            if 0 < value < 150: temps.append({'name':name+' / '+label,'celsius':value})
        except (OSError, ValueError): pass
    gpus = []
    fields = ['index','name','temperature.gpu','utilization.gpu','memory.used','memory.total','power.draw','power.limit']
    raw = command(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'])
    for row in csv.reader(raw.splitlines()):
        if len(row) != len(fields): continue
        record = dict(zip(fields,[x.strip() for x in row]))
        for k in fields:
            if k != 'name':
                try: record[k] = float(record[k])
                except ValueError: record[k] = None
        gpus.append(record)
    return {'timestamp':time.time(),'hostname':os.uname().nodename,'cpu_percent':usage('cpu'),'cpu_cores':[{'id':k,'percent':usage(k)} for k in b if k != 'cpu' and k in a],'cores':os.cpu_count(),'load':os.getloadavg(),'uptime':float(open('/proc/uptime').read().split()[0]),'memory':{'total':mem['MemTotal'],'available':mem.get('MemAvailable',mem.get('MemFree',0)),'swap_total':mem['SwapTotal'],'swap_free':mem['SwapFree']},'network':net,'temperatures':temps,'gpus':gpus}

def slow_snapshot():
    gpu_processes = []
    raw = command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
    for row in csv.reader(raw.splitlines()):
        if len(row)==3: gpu_processes.append(dict(zip(['pid','name','memory_mib'],[x.strip() for x in row])))
    containers = []
    raw = command(['docker','ps','--format','{{json .}}'])
    for line in raw.splitlines()[:30]:
        try: c = json.loads(line)
        except ValueError: continue
        c['processes'] = command(['docker','top',c['ID'],'-eo','pid,ppid,etime,pcpu,pmem,args']).strip()
        containers.append(c)
    raw_stats = command(['docker','stats','--no-stream','--format','{{json .}}'],timeout=8) if containers else ''
    stats = {}
    for line in raw_stats.splitlines():
        try:
            item = json.loads(line); stats[item['Name']] = item
        except (ValueError, KeyError): pass
    for c in containers: c['stats'] = stats.get(c['Names'],{})
    mounts = []
    for mount in ['/', '/home', '/dev/shm']:
        d = shutil.disk_usage(mount)
        mounts.append({'mount':mount,'total':d.total,'used':d.used,'free':d.free})
    return {'details_timestamp':time.time(),'gpu_processes':gpu_processes,'containers':containers,'disks':mounts,'top_processes':command(['ps','-eo','pid,user,etime,pcpu,pmem,args','--sort=-pcpu']).splitlines()[:21]}

