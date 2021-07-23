'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class MyWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }
    
   
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {

     
	 await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);

        this.workerIndex = workerIndex;
        this.totalWorkers = totalWorkers;
        this.roundIndex = roundIndex;
        this.roundArguments = roundArguments;
        this.sutAdapter = sutAdapter;
        this.sutContext = sutContext;
    }
    
    async submitTransaction() {
        
	console.log(`Hash of picture:${this.workerIndex}`)
	const request = {
		contractId: 'asn1',
        	contractFunction: 'UploadPhoto',
            	invokerIdentity: 'Admin@org2.example.com',
            	contractArguments: ['OA','OSN1','PointerPA','PointerPolivyP',this.workerIndex],
            	readOnly: false
	};	
	await this.sutAdapter.sendRequests(request);
        
    }
    
    async cleanupWorkloadModule() {
       
    }
    

    

  
}



function createWorkloadModule() {
    return new MyWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;

