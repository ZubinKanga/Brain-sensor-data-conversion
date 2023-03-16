/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.3.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../eego2lsl/mainwindow.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.3.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Reader_t {
    const uint offsetsAndSize[34];
    char stringdata0[152];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(offsetof(qt_meta_stringdata_Reader_t, stringdata0) + ofs), len 
static const qt_meta_stringdata_Reader_t qt_meta_stringdata_Reader = {
    {
QT_MOC_LITERAL(0, 6), // "Reader"
QT_MOC_LITERAL(7, 8), // "finished"
QT_MOC_LITERAL(16, 0), // ""
QT_MOC_LITERAL(17, 7), // "timeout"
QT_MOC_LITERAL(25, 11), // "ampNotFound"
QT_MOC_LITERAL(37, 14), // "connectionLost"
QT_MOC_LITERAL(52, 12), // "unknownError"
QT_MOC_LITERAL(65, 4), // "read"
QT_MOC_LITERAL(70, 7), // "setStop"
QT_MOC_LITERAL(78, 4), // "stop"
QT_MOC_LITERAL(83, 9), // "setParams"
QT_MOC_LITERAL(93, 5), // "capId"
QT_MOC_LITERAL(99, 12), // "samplingRate"
QT_MOC_LITERAL(112, 8), // "BipRange"
QT_MOC_LITERAL(121, 8), // "EegRange"
QT_MOC_LITERAL(130, 10), // "hexEegMask"
QT_MOC_LITERAL(141, 10) // "hexBipMask"

    },
    "Reader\0finished\0\0timeout\0ampNotFound\0"
    "connectionLost\0unknownError\0read\0"
    "setStop\0stop\0setParams\0capId\0samplingRate\0"
    "BipRange\0EegRange\0hexEegMask\0hexBipMask"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Reader[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   62,    2, 0x06,    1 /* Public */,
       3,    0,   63,    2, 0x06,    2 /* Public */,
       4,    0,   64,    2, 0x06,    3 /* Public */,
       5,    0,   65,    2, 0x06,    4 /* Public */,
       6,    0,   66,    2, 0x06,    5 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       7,    0,   67,    2, 0x0a,    6 /* Public */,
       8,    1,   68,    2, 0x0a,    7 /* Public */,
      10,    6,   71,    2, 0x0a,    9 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    9,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Double, QMetaType::Double, QMetaType::ULongLong, QMetaType::ULongLong,   11,   12,   13,   14,   15,   16,

       0        // eod
};

void Reader::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Reader *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->finished(); break;
        case 1: _t->timeout(); break;
        case 2: _t->ampNotFound(); break;
        case 3: _t->connectionLost(); break;
        case 4: _t->unknownError(); break;
        case 5: _t->read(); break;
        case 6: _t->setStop((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 7: _t->setParams((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast< std::add_pointer_t<double>>(_a[4])),(*reinterpret_cast< std::add_pointer_t<qulonglong>>(_a[5])),(*reinterpret_cast< std::add_pointer_t<qulonglong>>(_a[6]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Reader::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Reader::finished)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Reader::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Reader::timeout)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Reader::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Reader::ampNotFound)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Reader::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Reader::connectionLost)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Reader::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Reader::unknownError)) {
                *result = 4;
                return;
            }
        }
    }
}

const QMetaObject Reader::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_Reader.offsetsAndSize,
    qt_meta_data_Reader,
    qt_static_metacall,
    nullptr,
qt_incomplete_metaTypeArray<qt_meta_stringdata_Reader_t
, QtPrivate::TypeAndForceComplete<Reader, std::true_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>
, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<bool, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<int, std::false_type>, QtPrivate::TypeAndForceComplete<int, std::false_type>, QtPrivate::TypeAndForceComplete<double, std::false_type>, QtPrivate::TypeAndForceComplete<double, std::false_type>, QtPrivate::TypeAndForceComplete<unsigned long long, std::false_type>, QtPrivate::TypeAndForceComplete<unsigned long long, std::false_type>


>,
    nullptr
} };


const QMetaObject *Reader::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Reader::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Reader.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Reader::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void Reader::finished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void Reader::timeout()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void Reader::ampNotFound()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void Reader::connectionLost()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void Reader::unknownError()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}
struct qt_meta_stringdata_MainWindow_t {
    const uint offsetsAndSize[26];
    char stringdata0[151];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs), len 
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 10), // "MainWindow"
QT_MOC_LITERAL(11, 14), // "threadFinished"
QT_MOC_LITERAL(26, 0), // ""
QT_MOC_LITERAL(27, 13), // "threadTimeout"
QT_MOC_LITERAL(41, 14), // "connectionLost"
QT_MOC_LITERAL(56, 11), // "ampNotFound"
QT_MOC_LITERAL(68, 12), // "unknownError"
QT_MOC_LITERAL(81, 18), // "load_config_dialog"
QT_MOC_LITERAL(100, 18), // "save_config_dialog"
QT_MOC_LITERAL(119, 4), // "link"
QT_MOC_LITERAL(124, 10), // "closeEvent"
QT_MOC_LITERAL(135, 12), // "QCloseEvent*"
QT_MOC_LITERAL(148, 2) // "ev"

    },
    "MainWindow\0threadFinished\0\0threadTimeout\0"
    "connectionLost\0ampNotFound\0unknownError\0"
    "load_config_dialog\0save_config_dialog\0"
    "link\0closeEvent\0QCloseEvent*\0ev"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   68,    2, 0x08,    1 /* Private */,
       3,    0,   69,    2, 0x08,    2 /* Private */,
       4,    0,   70,    2, 0x08,    3 /* Private */,
       5,    0,   71,    2, 0x08,    4 /* Private */,
       6,    0,   72,    2, 0x08,    5 /* Private */,
       7,    0,   73,    2, 0x08,    6 /* Private */,
       8,    0,   74,    2, 0x08,    7 /* Private */,
       9,    0,   75,    2, 0x08,    8 /* Private */,
      10,    1,   76,    2, 0x08,    9 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 11,   12,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->threadFinished(); break;
        case 1: _t->threadTimeout(); break;
        case 2: _t->connectionLost(); break;
        case 3: _t->ampNotFound(); break;
        case 4: _t->unknownError(); break;
        case 5: _t->load_config_dialog(); break;
        case 6: _t->save_config_dialog(); break;
        case 7: _t->link(); break;
        case 8: _t->closeEvent((*reinterpret_cast< std::add_pointer_t<QCloseEvent*>>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainWindow.offsetsAndSize,
    qt_meta_data_MainWindow,
    qt_static_metacall,
    nullptr,
qt_incomplete_metaTypeArray<qt_meta_stringdata_MainWindow_t
, QtPrivate::TypeAndForceComplete<MainWindow, std::true_type>
, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<QCloseEvent *, std::false_type>


>,
    nullptr
} };


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 9;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
