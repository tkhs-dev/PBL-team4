package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.data.dao.Models
import ac.osaka_u.ics.pbl.data.entity.ModelEntity
import ac.osaka_u.ics.pbl.data.entity.toEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.Model
import org.jetbrains.exposed.sql.transactions.transaction
import java.io.File
import java.util.*

interface ModelRepository {
    fun findModelById(id: UUID): Model?
    fun findModelByTaskId(taskId: UUID): Model?
    fun findModels(): List<Model>
    fun createModel(model: Model): Model
    fun saveModelFile(id: UUID, modelByteArray: ByteArray)
}

class ModelRepositoryImpl(private val staticRootPath:String) : ModelRepository {
    init {
        val dir = File(staticRootPath)
        if(!dir.exists()) {
            dir.mkdirs()
        }
    }
    override fun findModelById(id: UUID): Model? {
        return transaction {
            ModelEntity.findById(id)?.toModel()
        }
    }

    override fun findModelByTaskId(taskId: UUID): Model? {
        return transaction {
            ModelEntity.find { Models.taskId eq taskId }.firstOrNull()?.toModel()
        }
    }

    override fun findModels(): List<Model> {
        return transaction {
            ModelEntity.all().map { it.toModel() }
        }
    }

    override fun createModel(model: Model): Model {
        return transaction {
            model.toEntity().toModel()
        }
    }

    override fun saveModelFile(id: UUID, modelByteArray: ByteArray) {
        val path = "$staticRootPath/models/$id"
        val file = File(path)
        if(!file.exists()) {
            file.createNewFile()
        }
        file.writeBytes(modelByteArray)
    }
}