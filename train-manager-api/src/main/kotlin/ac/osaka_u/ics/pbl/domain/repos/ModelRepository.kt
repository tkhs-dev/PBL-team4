package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.data.entity.ModelEntity
import ac.osaka_u.ics.pbl.data.entity.toEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.Model
import org.jetbrains.exposed.sql.transactions.transaction
import java.util.*

interface ModelRepository {
    fun findModelById(id: UUID): Model?
    fun findModels(): List<Model>
    fun createModel(model: Model): Model
}

class ModelRepositoryImpl : ModelRepository {
    override fun findModelById(id: UUID): Model? {
        return transaction {
            ModelEntity.findById(id)?.toModel()
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
}