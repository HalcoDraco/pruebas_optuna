Vagrant.configure("2") do |config|
  config.vm.define "container" do |c|
    c.vm.box = "sylabs/singularity-ce-3.9-ubuntu-bionic64"  # Replace YOUR_BOX_NAME with the actual box you're using (e.g., "hashicorp/bionic64")
    c.vm.provider "virtualbox" do |vb|
      vb.name = "container"
      vb.memory = "24576"
    end

    # Specify disk size here
    c.vm.disk :disk, size: "70GB", primary: true
  end
end
