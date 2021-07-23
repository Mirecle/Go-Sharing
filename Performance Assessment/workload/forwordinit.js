'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class MyWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }
    
   
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {

     
	 await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
	for (let i=0;i<this.roundArguments.assets;i++){
		const HashP = `${this.workerIndex}_${i}`;
		console.log(`Hash of picture:${HashP}`)
		const request = {
			contractId: 'asn1',
            	contractFunction: 'UploadPhoto',
            	invokerIdentity: 'Admin@org2.example.com',
            	contractArguments: ['OA','OSN1','PointerPA','PointerPolivyP',HashP],
            	readOnly: false
		};	
		await this.sutAdapter.sendRequests(request);
	}
        
    }
    
    async submitTransaction() {
        
	
        
    }
    
    async cleanupWorkloadModule() {
       
    }
    

    

  
}



function createWorkloadModule() {
    return new MyWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;

