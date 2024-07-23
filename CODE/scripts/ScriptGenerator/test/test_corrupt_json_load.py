import json

t = "/nfs2/harmonization/BIDS/HABSHD/sub-6561/ses-baseline/dwi/sub-6561_ses-baseline_dwi.json"

def find_problematic_line(t):
    with open(t, 'r') as f:
        #print the lines one by one
        for i,line in enumerate(f):
            try:
                print(line)
                print('k')
            except:
                #print("Error on line: ", line)
                return i
                pass

p = find_problematic_line(t)
print()
print('hi')
print(p)