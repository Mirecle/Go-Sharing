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
	 
     	  console.log(`Worker ${this.workerIndex}`);
        const myArgs = {
            contractId: 'mycc2',
            contractFunction: 'invoke',
            invokerIdentity: 'Admin@org1.example.com',
            contractArguments: ['a','b','20'],
            readOnly: false
        };
        await this.sutAdapter.sendRequests(myArgs);
	 
        
    }
    
    async cleanupWorkloadModule() {
       
    }
    

    

  
}



function createWorkloadModule() {
    return new MyWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;

