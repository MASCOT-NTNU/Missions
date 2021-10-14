# Here comes notes for using raspberry pi 4

`192.168.1.2`

`arp -a` show all the ip addresses in the same host
`touch newfile` will create a new empty file named after "newfile"
`ping raspberrypi.local` shows ip address
`passwd` changes the password
`raspberry` default password for the first time user
`hostname -I` shows ip addr of raspberry pi 4
`ssh` add empty file name in the root directory, then enables ssh




---

# If ~/.inputrc doesn't exist yet: First include the original /etc/inputrc
# so it won't get overriden
if [ ! -a ~/.inputrc ]; then echo '$include /etc/inputrc' > ~/.inputrc; fi

# Add shell-option to ~/.inputrc to enable case-insensitive tab completion
echo 'set completion-ignore-case On' >> ~/.inputrc
