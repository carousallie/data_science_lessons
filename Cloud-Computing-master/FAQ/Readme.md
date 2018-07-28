# FAQ

## Cloud installation errors and warnings :

1. IF I get an *SSH error saying the connection is not secure* error or warning while connecting to your VM (If you delete a VM instance that you have already connected to through SSH, 
and then create a new one that happens to have the same connection IP address), what do I need to do?. (Relevant to Mac users )

	+ Answer: To fix this, you need to open the "known_hosts"  text file in the ~/.ssh directory and delete the entry with the matching IP address. You can then connect again.
	
1. IF I get an *Permission denied* error or warning while connecting to your VM what do I need to do?. (Relevant to Mac users and windows users)

	+ Answer: First be sure you did put the public key on the ssh section while creating your instance. Then you need to be sure you used the same private key that you created while you used your public key.
	          Second you need to change the mode of public key to read only by enterin chmod 400 <your public key name>

