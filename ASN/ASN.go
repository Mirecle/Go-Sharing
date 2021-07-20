/*
 * @Author: ChZheng
 * @Date: 2021-01-07 17:17:27
 * @LastEditTime: 2021-07-19 16:24:14
 * @LastEditors: ChZheng
 * @Description:新版 ASN 链码
 * @FilePath: /Go-Sharing/ASN/ASN.go
 */
//TODO:命名规范
//TODO:白名单黑名单
//TODO:queryState
package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// ASNChaincode implementation
type ASNChaincode struct{}

func main() {
	err := shim.Start(new(ASNChaincode))
	if err != nil {
		fmt.Printf("Error starting Simple chaincode: %s", err)
	}
}

// Init initializes the chaincode
func (t *ASNChaincode) Init(stub shim.ChaincodeStubInterface) peer.Response {
	policyPointer := "PointerPolicyP"
	//"AGp,DGp,RGp,RUGp"
	policyP := "v1-v2-v3-v4-v5*v1-v2-v3-v4*v1-v2-v3*v1-v2"

	err := stub.PutState(policyPointer, []byte(policyP))
	if err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(nil)
}

// Invoke functions
func (t *ASNChaincode) Invoke(stub shim.ChaincodeStubInterface) peer.Response {
	fun, args := stub.GetFunctionAndParameters()
	if fun == "UploadPhoto" {
		return uploadPhoto(stub, args)
	} else if fun == "VisitPhoto" {
		return visitOrDownloadPhoto(stub, args, "Visiting")
	} else if fun == "DownloadPhoto" {
		return visitOrDownloadPhoto(stub, args, "Downloading")
	} else if fun == "ForwardPhoto" {
		return forwardOrReuploadPhoto(stub, args, "Forwarding")
	} else if fun == "ReuploadPhoto" {
		return forwardOrReuploadPhoto(stub, args, "Reuploading")
	} else if fun == "DeletePhoto" {
		return deletePhoto(stub, args)
	}
	return shim.Error("Commend is not defined")
}

/**
 * @description:peer chaincode invoke -n mycc -c '{"Args":["UploadPhoto","OA","OSN1","PointerPA","v1-v2-v3-v4-v5*v1-v2-v3-v4*v1-v2-v3*v1-v2","HashPA"]}' -C myc
 * @param {shim.ChaincodeStubInterface} stub
 * @param {[]string} args
 * @return {*}
 */
func uploadPhoto(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	UserID := args[0]
	OSNID := args[1]
	PointerPA := args[2]
	PointerPolicyP := args[3]
	HashPA := args[4]

	if checkExistencePhoto(stub, HashPA, UserID, OSNID) {
		return shim.Error("picture has already been uploaded before,please use cammand reupload") // TODO：直接调用二次上传
	}
	policyPointer := "PointerPolicyP"
	err := stub.PutState(policyPointer, []byte(PointerPolicyP))
	if err != nil {
		return shim.Error(err.Error())
	}

	SMCaddr := createSMC(stub, HashPA, UserID, OSNID, PointerPA, PointerPolicyP)
	PMCC := UserID + "," + OSNID + "," + SMCaddr
	err = stub.PutState(HashPA, []byte(PMCC))
	if err != nil {
		return shim.Error(err.Error())
	}

	return shim.Success([]byte(UserID + "." + OSNID + " Upload " + HashPA + " Success ")) //回显用于测试
	//return shim.Success(nil)

}

/**
 * @description:peer chaincode invoke -n mycc -c '{"Args":["VisitPhoto","v2","OSN2","OA","OSN1","HashPA"]}' -C myc
 * @param {shim.ChaincodeStubInterface} stub
 * @param {[]string} args
 * @param {string} OperationT
 * @return {*}
 */
func visitOrDownloadPhoto(stub shim.ChaincodeStubInterface, args []string, OperationT string) peer.Response {
	VB := args[0]
	OSN2 := args[1]
	OA := args[2]
	OSN1 := args[3]
	HashPA := args[4]

	error := checkAccPro(stub, HashPA, OA, OSN1)
	if error != "nil" {
		return shim.Error(error)
	}

	Pid := "PS" + HashPA + OA + OSN1
	ProChain, _ := stub.GetState(Pid)
	fmt.Println("visitOrDownloadPhoto ProChain------->" + string(ProChain))
	if ProChain == nil {
		return shim.Error("ProChain err1")
	}
	ProChainStr := getStrings(string(ProChain), '+')
	if len(ProChainStr) != 5 {
		return shim.Error("ProChain err2")
	}
	PolicyPointer := ProChainStr[2]
	//PolicyP, _ := stub.GetState(PolicyPointer) //其实是存在数据库中的
	PolicyP := PolicyPointer
	fmt.Println("visitOrDownloadPhoto Policy-------->" + string(PolicyP))
	result := checkPolicy(string(PolicyP), OperationT, VB)

	addAccessRecord(stub, HashPA, VB, OSN2, OperationT, result)

	return shim.Success([]byte(VB + "." + OSN2 + " visit " + OA + "." + OSN1 + " " + result))
	//return shim.Success(nil)
}

/**
 * @description:peer chaincode invoke -n mycc -c '{"Args":["ForwordPhoto","v2","OSN3","v1-v2-v3-v4-v5
*v1-v2-v3-v4*v1-v2-v3*v1","OA","OSN1","HashPA"]}' -C myc
 * @param {shim.ChaincodeStubInterface} stub
 * @param {[]string} args
 * @param {string} OperationT
 * @return {*}
*/
func forwardOrReuploadPhoto(stub shim.ChaincodeStubInterface, args []string, OperationT string) peer.Response {
	VC := args[0]
	OSN3 := args[1]
	VCpolicyP := args[2]
	OA := args[3]
	OSN1 := args[4]
	HashPA := args[5]

	error := checkAccPro(stub, HashPA, OA, OSN1)
	if error != "nil" {
		return shim.Error(error)
	}

	Pid := "PS" + HashPA + OA + OSN1
	ProChain, _ := stub.GetState(Pid)
	fmt.Println("forwardOrReuploadPhoto ProChain-------->" + string(ProChain))
	if ProChain == nil {
		return shim.Error("ProChain err1")
	}
	ProChainStr := getStrings(string(ProChain), '+')
	if len(ProChainStr) != 5 {
		return shim.Error("ProChain err2")
	}
	PolicyPointer := ProChainStr[2]
	//VApolicyP, _ := stub.GetState(PolicyPointer)
	VApolicyP := PolicyPointer
	fmt.Println("forwardOrReuploadPhoto VApolicyP-------->" + string(VApolicyP))
	result := checkPolicy(string(VApolicyP), OperationT, VC) //未加入组织权限
	if result == "refuse" {
		return shim.Error("The propagation state of " + OSN3 + "." + VC + " has no right to operate OperationT’")
	}
	SMCaddr := "S" + HashPA + OA + OSN1
	PolicyNew := createPolicy(string(VApolicyP), VCpolicyP)
	addAccessRecord(stub, HashPA, VC, OSN3, OperationT, result)
	addPropagationChain(stub, HashPA, SMCaddr, VC, OSN3, "PointerPANew", PolicyNew)

	return shim.Success([]byte(VC + "." + OSN3 + " forward " + OA + "." + OSN1 + " " + result))
	//return shim.Success(nil)

}

/**
 * @description:peer chaincode invoke -n mycc -c '{"Args":["DeletePhoto","OA","OSN1","HashPA"]}' -C myc
 * @param {shim.ChaincodeStubInterface} stub
 * @param {[]string} args
 * @return {*}
 */
func deletePhoto(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	UserID := args[0]
	OSNID := args[1]
	HashPA := args[2]
	OperationT := "Delete"
	result := "adopt"

	err := checkAccPro(stub, HashPA, UserID, OSNID)
	if err != "nil" {
		return shim.Error(err)
	}
	SMCaddr := "S" + HashPA + UserID + OSNID
	PidChainID := "CP" + SMCaddr

	delNode(stub, PidChainID)

	addAccessRecord(stub, HashPA, UserID, OSNID, OperationT, result)
	return shim.Success([]byte(UserID + "." + OSNID + " delete " + HashPA + " " + result))
	//return shim.Success(nil)
}

func checkExistencePhoto(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNID string) bool {
	result, _ := stub.GetState(HashPA)
	fmt.Println("chechExistencePhoto result-------->" + string(result))
	if result == nil {
		return false
	}
	Pid := "PS" + HashPA + UserID + OSNID
	ProChain, _ := stub.GetState(Pid)
	fmt.Println("chechExistencePhoto ProChain-------->" + string(ProChain))
	if ProChain == nil {
		return false
	}
	ProChainStr := getStrings(string(ProChain), '+')
	if len(ProChainStr) != 5 {
		return false
	}
	Status := ProChainStr[4]
	return Status == "Active" //it is bool
}

/**
 * @description:生成访问记录
 * @param {shim.ChaincodeStubInterface} stub
 * @param {string} HashPA
 * @param {string} UserID
 * @param {string} OSNId
 * @param {string} OperationT
 * @param {string} result
 * @return {*}
 */
func addAccessRecord(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNId string, OperationT string, result string) (string, string) {
	IDARL := "I" + HashPA
	var Pid string
	Timestamp := time.Now().Format("20060102150405")
	if OperationT == "Visiting" || OperationT == "Downloading" || OperationT == "Delete" || result == "refuse" {
		Pid = "0"
	} else {
		Pid = "PS" + HashPA + UserID + OSNId
	}
	var AccRecList string
	if OperationT == "Uploading" {
		AccRecList = UserID + "+" + OSNId + "+" + OperationT + "+" + result + "+" + Timestamp + "+" + Pid
	} else {
		ARL, _ := stub.GetState(IDARL)
		fmt.Println("addAccessRecord ARL-------->" + string(ARL))
		AccRecList = string(ARL) + "," + UserID + "+" + OSNId + "+" + OperationT + "+" + result + "+" + Timestamp + "+" + Pid
	}
	fmt.Println("addAccessRecord AccRecList--------> " + AccRecList)
	err := stub.PutState(IDARL, []byte(AccRecList))
	if err != nil {
		IDARL = err.Error()
	}
	return IDARL, AccRecList
}

/**
 * @description:生成传播链
 * @param {shim.ChaincodeStubInterface} stub
 * @param {string} HashPA
 * @param {string} SMCaddr
 * @param {string} UserID
 * @param {string} OSNID
 * @param {string} PhotoAddress
 * @param {string} PolicyAddress
 * @return {*}
 */
func addPropagationChain(stub shim.ChaincodeStubInterface, HashPA string, SMCaddr string, UserID string, OSNID string, PhotoAddress string, PolicyAddress string) string {
	PidChainID := "CP" + SMCaddr
	Pid := "PS" + HashPA + UserID + OSNID
	FrontID := "P" + SMCaddr //第一次上传指向自己，二次上传或转发指向前一个节点
	// when test zhushi this
	PRC, _ := stub.GetState(Pid)
	if PRC != nil {
		fmt.Println("addPropagationChain PRC-------->" + string(PRC))
		PRCstr := getStrings(string(PRC), '+')
		if PRCstr[4] != "Inactive" {
			return PidChainID
		}
	}

	var ProChain, Status, PidChain string
	Status = "Active"
	PidC, _ := stub.GetState(PidChainID) //同源图片在一条pid 链上
	fmt.Println("addPropagationChain PidChain1-------->" + string(PidC))
	if PidC == nil {
		PidChain = "P" + SMCaddr + "," + Pid
	} else {
		PidChain = string(PidC) + "," + Pid
	}
	fmt.Println("addPropagationChain PidChain2--------> " + PidChain)
	ProChain = Pid + "+" + PhotoAddress + "+" + PolicyAddress + "+" + FrontID + "+" + Status
	fmt.Println("addPropagationChain ProChain--------> " + ProChain)
	err2 := stub.PutState(PidChainID, []byte(PidChain)) //Pid 链
	if err2 != nil {
		return "PidChainError" + err2.Error()
	}
	err3 := stub.PutState(Pid, []byte(ProChain)) //传播链
	if err3 != nil {
		return "ProChainError" + err3.Error()
	}
	return PidChainID
}

/**
 * @description:生成初试 SMC
 * @param {shim.ChaincodeStubInterface} stub
 * @param {string} HashPA
 * @param {string} UserID
 * @param {string} OSNID
 * @param {string} PhotoAddress
 * @param {string} PolicyAddress
 * @return {*}
 */
func createSMC(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNID string, PhotoAddress string, PolicyAddress string) string {
	SMCaddr := "S" + HashPA + UserID + OSNID
	OperationT := "Uploading"
	Result := " "
	IDARL, _ := addAccessRecord(stub, HashPA, UserID, OSNID, OperationT, Result)
	PidChainID := addPropagationChain(stub, HashPA, SMCaddr, UserID, OSNID, PhotoAddress, PolicyAddress)
	SMCC := IDARL + "," + PidChainID

	err := stub.PutState(SMCaddr, []byte(SMCC)) //SMC
	if err != nil {
		return "SMCaddrError" + err.Error()
	}
	return SMCaddr
}

/**
 * @description: 通过 key 从区块链中获取值（字符串格式）并以","为界切割字符串
 * @param {shim.ChaincodeStubInterface} stub
 * @param {string} Key
 * @return {*}
 */
func getValues(stub shim.ChaincodeStubInterface, Key string) []string {
	value, _ := stub.GetState(Key)
	fmt.Println("getValues value-------->" + string(value))
	if value == nil {
		return nil
	}
	valueStr := getStrings(string(value), ',')
	return valueStr
}

/**
 * @description:以指定字符为界分割字符串
 * @param {string} str
 * @param {rune} sym
 * @return {*}
 */
func getStrings(str string, sym rune) []string {
	f := func(c rune) bool {
		return c == sym
	}
	strs := strings.FieldsFunc(str, f)
	return strs
}

/**
 * @description:对上传图片进行一系列检查
 * @param {shim.ChaincodeStubInterface} stub
 * @param {string} HashPA
 * @param {string} UserID
 * @param {string} OSNId
 * @return {*}
 */
func checkAccPro(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNId string) string {
	var err string
	err = "nil"
	if !checkExistencePhoto(stub, HashPA, UserID, OSNId) {
		err = "There is no picture in Picture pool matched " + HashPA
		return err
	}

	IDARL := "I" + HashPA
	Pid := "PS" + HashPA + UserID + OSNId
	AccRecList, _ := stub.GetState(IDARL)
	fmt.Println("checkAccPro AccRecList-------->" + string(AccRecList))
	if AccRecList == nil {
		err = OSNId + "." + UserID + " is an illegal Photo owner"
		return err
	}
	ProChain, _ := stub.GetState(Pid)
	fmt.Println("checkAccPro ProChain-------->" + string(ProChain))
	if ProChain == nil {
		err = "err ProChain1"
		return err

	}
	ProChainStr := getStrings(string(ProChain), '+')
	if len(ProChainStr) != 5 {
		err = "err ProChain2"
		return err
	}
	Status := ProChainStr[4]
	if Status != "Active" {
		err = "The propagation state of " + OSNId + "." + UserID + " is not active"
		return err
	}
	return err
}

/**
 * @description:检查隐私策略是否符合
 * @param {string} PolicyP
 * @param {string} OperationT
 * @param {string} PhotoVisitor
 * @return {*}
 */
func checkPolicy(PolicyP string, OperationT string, PhotoVisitor string) string {
	PolicyPstr := getStrings(PolicyP, '*')
	if len(PolicyPstr) == 0 {
		return "refuse"
	}
	AGpstr := getStrings(PolicyPstr[0], '-')
	DGpstr := getStrings(PolicyPstr[1], '-')
	RGpstr := getStrings(PolicyPstr[2], '-')
	RUGpstr := getStrings(PolicyPstr[3], '-')

	if len(AGpstr) == 0 || len(DGpstr) == 0 || len(RGpstr) == 0 || len(RUGpstr) == 0 {
		return "refuse1"
	}
	if OperationT == "Visiting" {
		for _, v := range AGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}
		}
	} else if OperationT == "Downloading" {
		for _, v := range DGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}

		}
	} else if OperationT == "Forwarding" {
		for _, v := range RGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}
		}
	} else if OperationT == "Reuploading" {
		for _, v := range RUGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}

		}
	}

	return "refuse"
}

/**
 * @description:合并隐私策略 1 和 2
 * @param {string} PolicyP1
 * @param {string} PolicyP2
 * @return {*}
 */
func createPolicy(PolicyP1 string, PolicyP2 string) string {
	PolicyPstr1 := getStrings(PolicyP1, '*')
	if len(PolicyPstr1) == 0 {
		return "refuse"
	}
	AGpstr1 := getStrings(PolicyPstr1[0], '-')
	DGpstr1 := getStrings(PolicyPstr1[1], '-')
	RGpstr1 := getStrings(PolicyPstr1[2], '-')
	RUGpstr1 := getStrings(PolicyPstr1[3], '-')

	if len(AGpstr1) == 0 || len(DGpstr1) == 0 || len(RGpstr1) == 0 || len(RUGpstr1) == 0 {
		return "refuse"
	}
	PolicyPstr2 := getStrings(PolicyP2, '*')
	if len(PolicyPstr2) == 0 {
		return "refuse"
	}
	AGpstr2 := getStrings(PolicyPstr2[0], '-')
	DGpstr2 := getStrings(PolicyPstr2[1], '-')
	RGpstr2 := getStrings(PolicyPstr2[2], '-')
	RUGpstr2 := getStrings(PolicyPstr2[3], '-')

	if len(AGpstr2) == 0 || len(DGpstr2) == 0 || len(RGpstr2) == 0 || len(RUGpstr2) == 0 {
		return "refuse"
	}
	AGp3 := intersection(AGpstr1, AGpstr2)
	DGp3 := intersection(DGpstr1, DGpstr2)
	RGp3 := intersection(RGpstr1, RGpstr2)
	RUGp3 := intersection(RUGpstr1, RUGpstr2)
	PolicyP3 := AGp3 + "*" + DGp3 + "*" + RGp3 + "*" + RUGp3
	return PolicyP3
}

/**
 * @description: 两个字符串取交集
 * @param {[]string} a
 * @param {[]string} b
 * @return {*}
 */
func intersection(a []string, b []string) (inter string) {
	// interacting on the smallest list first can potentailly be faster...but not by much,worse case is the same
	low, high := a, b
	if len(a) > len(b) {
		low = b
		high = a
	}

	done := false
	for i, l := range low {
		for j, h := range high {
			f1 := i + 1
			f2 := j + 1
			if l == h {
				inter = inter + "-" + h
				if f1 < len(low) && f2 < len(high) {

					if low[f1] != high[f2] {
						done = true
					}
				}

				high = high[:j+copy(high[j:], high[j+1:])]
				break
			}
		}
		if done {
			break
		}
	}
	return
}

/**
 * @description:递归删除以某个节点为首的整个分支
 * @param {shim.ChaincodeStubInterface} stub
 * @param {string} PidChainID
 * @return {*}
 */
func delNode(stub shim.ChaincodeStubInterface, PidChainID string) bool {
	PidStr := getValues(stub, PidChainID)
	if PidStr == nil {

		Pid := string([]byte(PidChainID)[1:])
		ProC, _ := stub.GetState(Pid)
		fmt.Println("delNode ProChain1-------->" + string(ProC))
		if ProC == nil {
			return false
		}
		fmt.Println(string(ProC))
		ProChainStr := getStrings(string(ProC), '+')
		var ProChain string
		if ProChainStr[4] != "Inactive" {
			ProChain = ProChainStr[0] + "+" + ProChainStr[1] +
				"+" + ProChainStr[2] + "+" + ProChainStr[3] + "+" + "Inactive"
		}
		fmt.Println("delNode ProChain2-------->" + string(ProChain))
		err2 := stub.PutState(Pid, []byte(ProChain)) //传播链
		if err2 != nil {
			return false
		}
		return false
	}
	for _, Pid := range PidStr {
		ProC, _ := stub.GetState(Pid)
		fmt.Println("delNode ProChain1-------->" + string(ProC))
		if ProC == nil {
			return false
		}
		fmt.Println(string(ProC))
		ProChainStr := getStrings(string(ProC), '+')
		var ProChain string
		if ProChainStr[4] != "Inactive" {
			ProChain = ProChainStr[0] + "+" + ProChainStr[1] +
				"+" + ProChainStr[2] + "+" + ProChainStr[3] + "+" + "Inactive"
		}
		fmt.Println("delNode ProChain2-------->" + string(ProChain))
		err2 := stub.PutState(Pid, []byte(ProChain)) //传播链
		if err2 != nil {
			return false
		}
		if "C"+Pid == PidChainID {
			continue
		}
		delNode(stub, "C"+Pid)
	}
	return true
}
