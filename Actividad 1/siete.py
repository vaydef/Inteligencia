#siete

i = 1
while True:
    if (i+3)%10==0:
        print("clap", end=", ")
        i+=1
    if i%7==0:
        if i==70:
            while i<80:
                print("clap", end=", ")
                i+=1
        else:
            print("clap", end=", ")
            i+=1
    if i==100:
        print(i, end=".")
        break
    print(i, end=", ")
    i+=1
