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
        const hash = new Date().getTime().toString();
	   const padding= this.workerIndex.toString();
        console.log(`Worker ${this.workerIndex}:sutAdapter"`+hash+padding);
        const myArgs = {
            contractId: 'asn1',
            contractFunction: 'VisitPhoto',
            invokerIdentity: 'Admin@org2.example.com',
            contractArguments: ['v9','OSN2','OA','OSN1',hash+padding],
            readOnly: true
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

    
