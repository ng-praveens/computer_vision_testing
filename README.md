# Proposal_creation_TAX
Proposal Card Creation from the user input in form of free text and for tax created some dynamically questions to answer and from that proposal has created. Alonng with that dynamic pricing aslo added 
Deployed using AWS EC2  - Used Ubuntu in EC2.

# expertDP - Expert Dynamic profile generation using streamlit applications


Conversational_matching_proposal_withprice.py - main file with price added
Conversational_matching_proposals - main file without price

# How to Deploy Streamlit app on EC2 instance

## 1. Login with your AWS console and launch an EC2 instance

## 2. Run the following commands

### Note: Do the port mapping to this port:- 8501

```bash
sudo apt update
```

```bash
sudo apt-get update
```

```bash
sudo apt upgrade -y
```

```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```

```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```

```bash
git clone "Your-repository"
```

```bash
sudo apt install python3-pip
```

```bash
#Install the python3-venv package to enable the creation of virtual environments.
sudo apt-get install python3-venv
```

```bash
python3 -m venv myenv
source myenv/bin/activate

sudo apt install -y python3.10 python3.10-venv python3.10-distutils
installs Python 3.10 along with the necessary modules for creating virtual environments (python3.10-venv) and managing Python packages (python3.10-distutils) on a Linux system, with automatic confirmation of installation prompts (-y).

python3.10 -m venv myenv1



```

```bash
pip3 install -r requirements.txt
```

```bash
#Create the .env File
nano .env
#you need to create a .env file in your project directory on the EC2 instance. This file will hold your environment variables.
Save and exit the file by pressing CTRL + X, then Y, and ENTER.
```

```bash
#Temporary running
python3 -m streamlit run app.py
```

```bash
#Permanent running
nohup python3 -m streamlit run app.py
```

Note: Streamlit runs on this port: 8501
      if we add another updated file in the git repo then the new streamlit port : 8502 [ this will go on as we are addding new files] this should be added i the instance - security groups - edit securitygroups - edit inbound rules - tcp-8502-00000

```bash
#commands to use newly added py files and then pulling to ec2 instance
git fetch
git pull
```





